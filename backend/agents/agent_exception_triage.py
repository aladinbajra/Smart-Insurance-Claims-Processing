from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml

from backend.models.schemas import (
    ApprovalPacket,
    ExceptionEntry,
    Finding,
    RoutingDecision,
    RunMetrics,
    Severity,
)


# Deterministic severity ordering for finding prioritisation (REQ-027).
_SEVERITY_RANK: dict[Severity, int] = {
    Severity.critical: 0,
    Severity.high: 1,
    Severity.medium: 2,
    Severity.low: 3,
    Severity.info: 4,
}

_UPSTREAM_ARTIFACTS = (
    "context_packet.json",
    "claim_summary.json",
    "coverage_result.json",
    "settlement_calc.json",
    "fraud_score.json",
    "liability_result.json",
    "compliance_result.json",
    "anomaly_report.json",
)


class AgentExceptionTriage:
    """
    Agent H: Exception Triage and Lead Orchestrator (Orchestrator + Judge).
    REQ-026, REQ-027, REQ-028.

    Agent H is the terminal agent in the SICPS pipeline. It reads every
    upstream artifact from the shared run directory, merges and deduplicates
    findings from Agents A-E, derives a deterministic, rule-based routing
    decision, and writes the final run artifacts:

    - approval_packet.json  routing decision + evidence bundle (REQ-026)
    - exceptions.md         actionable exception summary (REQ-026)
    - audit_log.md          consolidated decision trace with evidence (REQ-045)
    - metrics.json          run metrics (REQ-029, REQ-042)

    All routing logic is driven by config/policy.yaml thresholds so that the
    same inputs always yield the same decision (idempotency, REQ-028) and a
    policy edit visibly changes behaviour without code changes (REQ-058).
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.policy = self._load_yaml(self.config_path)
        self.approval = self.policy.get("approval", {}) or {}
        self.cat_cfg = self.policy.get("cat_event", {}) or {}
        self.privacy = self.policy.get("privacy", {}) or {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(
        self,
        run_dir: str | Path,
        pipeline_duration_seconds: float = 0.0,
        agents_executed: list[str] | None = None,
        cache_hit: bool = False,
    ) -> dict[str, Any]:
        """
        Consolidate one claim run and produce the final artifacts.

        ``pipeline_duration_seconds``, ``agents_executed`` and ``cache_hit``
        are optional inputs supplied by the runner. They default to
        deterministic values so Agent H output stays byte-for-byte
        reproducible when run standalone (REQ-044).
        """

        run_path = Path(run_dir)

        context = self._require_json(run_path / "context_packet.json")

        # Optional upstream outputs — Agent H degrades gracefully if an agent
        # has not run, but in the full pipeline all of these exist.
        claim_summary = self._optional_json(run_path / "claim_summary.json")
        coverage = self._optional_json(run_path / "coverage_result.json")
        settlement = self._optional_json(run_path / "settlement_calc.json")
        fraud = self._optional_json(run_path / "fraud_score.json")
        # Stretch agents (advisory) — optional; H merges their findings if present.
        liability = self._optional_json(run_path / "liability_result.json")
        compliance = self._optional_json(run_path / "compliance_result.json")
        anomaly = self._optional_json(run_path / "anomaly_report.json")

        claim_id: str = context["claim_id"]
        run_id: str = str(context.get("run_id", claim_id))
        input_hash: str = str(context.get("input_hash", ""))
        finalized_at: str = str(context.get("created_at", "1970-01-01T00:00:00Z"))
        claim_type: str = str(context.get("claim_type", ""))
        flags: dict = context.get("flags", {}) or {}
        incident: dict = context.get("incident", {}) or {}

        # ----- Merge + dedupe + prioritise findings (REQ-027) -----
        findings = self._merge_findings(
            context, claim_summary, coverage, settlement, fraud,
            liability, compliance, anomaly,
        )

        # ----- Rule-based routing decision (REQ-027) -----
        routing, rationale = self._decide_routing(
            claim_type=claim_type,
            flags=flags,
            incident=incident,
            coverage=coverage,
            settlement=settlement,
            fraud=fraud,
            claim_summary=claim_summary,
        )

        net_settlement = self._final_net_settlement(routing, settlement)

        # ----- Exception categorisation (REQ-026) -----
        exceptions = self._build_exceptions(
            routing=routing,
            flags=flags,
            coverage=coverage,
            settlement=settlement,
            fraud=fraud,
            claim_summary=claim_summary,
            compliance=compliance,
            anomaly=anomaly,
        )

        evidence_bundle = self._evidence_bundle(findings, run_path)

        adjuster_notes = str(context.get("adjuster_notes", "") or "").strip() or rationale

        approval = ApprovalPacket(
            claim_id=claim_id,
            run_id=run_id,
            routing_decision=routing,
            net_settlement=net_settlement,
            adjuster_notes=adjuster_notes,
            exceptions=exceptions,
            all_findings=findings,
            evidence_bundle=evidence_bundle,
            input_hash=input_hash,
            finalized_at=finalized_at,
        )

        metrics = self._build_metrics(
            context=context,
            claim_summary=claim_summary,
            coverage=coverage,
            settlement=settlement,
            fraud=fraud,
            liability=liability,
            compliance=compliance,
            anomaly=anomaly,
            findings=findings,
            exceptions=exceptions,
            routing=routing,
            pipeline_duration_seconds=pipeline_duration_seconds,
            agents_executed=agents_executed,
            cache_hit=cache_hit,
        )

        # ----- Persist final artifacts -----
        self._write_json(
            run_path / "approval_packet.json", approval.model_dump(mode="json")
        )
        self._write_json(
            run_path / "metrics.json", metrics.model_dump(mode="json")
        )
        self._write_exceptions_md(
            run_path / "exceptions.md", claim_id, routing, net_settlement, exceptions
        )
        self._write_audit_log(
            run_path / "audit_log.md",
            claim_id=claim_id,
            run_id=run_id,
            input_hash=input_hash,
            finalized_at=finalized_at,
            routing=routing,
            net_settlement=net_settlement,
            rationale=rationale,
            findings=findings,
            exceptions=exceptions,
            evidence_bundle=evidence_bundle,
        )

        # CoreLogic-ready settlement payload in CSV (REQ-004).
        self._write_settlement_csv(
            run_path / "settlement_payload.csv",
            claim_id=claim_id,
            run_id=run_id,
            routing=routing,
            net_settlement=net_settlement,
            settlement=settlement,
            coverage=coverage,
            fraud=fraud,
            exception_count=len(exceptions),
            finalized_at=finalized_at,
            input_hash=input_hash,
        )

        # CoreLogic-ready settlement payload in JSON — the core-system schema
        # mapping (optional artifact #9). Same data as the CSV, structured for
        # posting to a downstream claims/settlement system.
        self._write_settlement_payload_json(
            run_path / "settlement_payload.json",
            claim_id=claim_id,
            run_id=run_id,
            routing=routing,
            net_settlement=net_settlement,
            settlement=settlement,
            coverage=coverage,
            fraud=fraud,
            exception_count=len(exceptions),
            finalized_at=finalized_at,
            input_hash=input_hash,
        )

        # Idempotent tracker work-item (REQ-051 stub: ADO/Jira posting). The
        # ticket id is derived from the claim id, so re-running the same claim
        # produces the same ticket — no duplicates.
        self._write_tracker_ticket(
            run_path / "tracker_ticket.json",
            claim_id=claim_id,
            run_id=run_id,
            routing=routing,
            net_settlement=net_settlement,
            exceptions=exceptions,
            adjuster_notes=adjuster_notes,
            finalized_at=finalized_at,
            input_hash=input_hash,
        )

        # Finalise Agent D's preliminary settlement (REQ-019/020).
        if settlement:
            settlement["finalized"] = True
            settlement["net_settlement"] = net_settlement
            self._write_json(run_path / "settlement_calc.json", settlement)

        return approval.model_dump(mode="json")

    # ------------------------------------------------------------------
    # Findings merge / dedupe / prioritise
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_findings(*sources: dict[str, Any]) -> list[Finding]:
        seen: set[str] = set()
        merged: list[Finding] = []

        for source in sources:
            for raw in source.get("findings", []) or []:
                finding_id = str(raw.get("finding_id", ""))
                if finding_id in seen:
                    continue
                seen.add(finding_id)
                merged.append(Finding.model_validate(raw))

        merged.sort(
            key=lambda f: (_SEVERITY_RANK.get(f.severity, 9), f.finding_id)
        )
        return merged

    # ------------------------------------------------------------------
    # Routing decision (the Judge)
    # ------------------------------------------------------------------

    def _decide_routing(
        self,
        claim_type: str,
        flags: dict,
        incident: dict,
        coverage: dict,
        settlement: dict,
        fraud: dict,
        claim_summary: dict,
    ) -> tuple[RoutingDecision, str]:
        """
        Deterministic first-match routing ladder. Order encodes priority:
        fraud and coverage denial are terminal; data-quality, total loss,
        late notice, duplicates and elevated fraud require a human; only a
        clean, covered, within-limit claim auto-settles.
        """

        # 1. Fraud — SIU referral is the most serious outcome.
        if fraud.get("siu_referral") is True:
            return (
                RoutingDecision.siu_referral,
                "Fraud screening triggered an SIU referral; settlement held at "
                "$0 pending investigation.",
            )

        # 2. Coverage denied (expired / pre-inception / exclusion).
        if coverage and coverage.get("coverage_active") is False:
            reason = coverage.get("denial_reason") or "coverage not active"
            return (
                RoutingDecision.coverage_denial,
                f"Coverage denied ({reason}); no settlement payable.",
            )

        # 3. Extraction quality too low to decide automatically.
        if claim_summary.get("requires_manual_review") is True:
            return (
                RoutingDecision.manual_review_route,
                "Extraction confidence below threshold (low OCR or missing "
                "documents); manual review required.",
            )

        # 4. Total loss requires market-value adjudication.
        if settlement.get("total_loss_flag") is True:
            return (
                RoutingDecision.total_loss_routing,
                "Total loss detected; route to total-loss market-value "
                "valuation.",
            )

        # 5. Late notice puts coverage at risk.
        if coverage.get("late_notice_violation") is True:
            return (
                RoutingDecision.late_notice_review,
                "Late notice violation; review reporting window against "
                "coverage before settlement.",
            )

        # 6. Possible duplicate submission.
        if flags.get("duplicate_claim") is True:
            return (
                RoutingDecision.adjuster_review,
                "Possible duplicate claim; adjuster verification required.",
            )

        # 7. Elevated (non-SIU) fraud indicators.
        if fraud.get("adjuster_review") is True:
            return (
                RoutingDecision.adjuster_review,
                "Elevated fraud indicators; adjuster review recommended.",
            )

        # 7b. Invoice/PO/GRN mismatch — billing variance approval.
        if settlement.get("invoice_match_status") == "mismatch":
            return (
                RoutingDecision.adjuster_review,
                "Invoice does not match PO/expected amount; billing variance "
                "approval required.",
            )

        # 8. Large gross exceeds the senior-review threshold.
        gross = self._as_float(settlement.get("gross_amount"))
        senior_threshold = float(
            self.approval.get("senior_review_threshold", 50000.0)
        )
        if settlement and gross > senior_threshold:
            return (
                RoutingDecision.senior_review,
                f"Gross amount {gross:,.2f} exceeds senior-review threshold "
                f"{senior_threshold:,.2f}; escalate to a senior adjuster.",
            )

        # 9. CAT surge fast-track — a live catastrophe event activates surge
        #    processing mode (distinct from straight-through auto-settle).
        if settlement:
            net = self._as_float(settlement.get("net_settlement"))

            surge_limit = float(self.cat_cfg.get("surge_auto_approve_limit", 0.0))
            if self._cat_surge_eligible(incident) and net <= surge_limit:
                return (
                    RoutingDecision.cat_surge_processing,
                    f"CAT surge processing mode: net {net:,.2f} within surge "
                    f"limit {surge_limit:,.2f}.",
                )

            home_limit = float(self.approval.get("homeowners_auto_approve_limit", 0.0))
            if claim_type == "homeowners" and net <= home_limit:
                return (
                    RoutingDecision.auto_settle,
                    f"Homeowners auto-approve: net {net:,.2f} within limit "
                    f"{home_limit:,.2f}.",
                )

            auto_limit = float(self.approval.get("auto_settle_max_amount", 0.0))
            if net <= auto_limit:
                return (
                    RoutingDecision.auto_settle,
                    f"Straight-through auto-settle: net {net:,.2f} within limit "
                    f"{auto_limit:,.2f}.",
                )

        # 10. Default — needs a human.
        return (
            RoutingDecision.adjuster_review,
            "Claim falls outside auto-settle limits; routed to adjuster review.",
        )

    def _cat_surge_eligible(self, incident: dict) -> bool:
        """
        Derive CAT surge eligibility from policy active_events (REQ-041) rather
        than trusting a precomputed flag — the incident must match a live event
        by event id or affected ZIP code while surge mode is enabled.
        """
        if not self.cat_cfg.get("surge_mode_enabled", False):
            return False
        if incident.get("cat_event") is not True:
            return False

        event_id = incident.get("cat_event_id")
        zip_code = str(incident.get("zip_code", ""))

        for event in self.cat_cfg.get("active_events", []) or []:
            if event_id and event_id == event.get("event_id"):
                return True
            affected = [str(z) for z in event.get("affected_zip_codes", []) or []]
            if zip_code and zip_code in affected:
                return True
        return False

    @staticmethod
    def _final_net_settlement(routing: RoutingDecision, settlement: dict) -> float:
        """SIU and coverage denial pay nothing; otherwise carry Agent D's net."""
        if routing in (RoutingDecision.siu_referral, RoutingDecision.coverage_denial):
            return 0.0
        return round(AgentExceptionTriage._as_float(settlement.get("net_settlement")), 2)

    # ------------------------------------------------------------------
    # Exception categorisation (REQ-026)
    # ------------------------------------------------------------------

    def _build_exceptions(
        self,
        routing: RoutingDecision,
        flags: dict,
        coverage: dict,
        settlement: dict,
        fraud: dict,
        claim_summary: dict,
        compliance: dict | None = None,
        anomaly: dict | None = None,
    ) -> list[ExceptionEntry]:
        exceptions: list[ExceptionEntry] = []

        def add(
            category: str,
            severity: Severity,
            description: str,
            recommended_action: str,
            evidence_links: list[str],
            assigned_to: str | None,
        ) -> None:
            exceptions.append(
                ExceptionEntry(
                    exception_id=f"EXC-{len(exceptions) + 1:03d}-{category.upper()}",
                    category=category,
                    severity=severity,
                    description=description,
                    recommended_action=recommended_action,
                    evidence_links=evidence_links,
                    assigned_to=assigned_to,
                )
            )

        if fraud.get("siu_referral") is True:
            ring = fraud.get("provider_ring_id")
            add(
                category="fraud_siu",
                severity=Severity.critical,
                description=(
                    "Fraud score crossed the SIU threshold"
                    + (f" (ring {ring})" if ring else "")
                    + "."
                ),
                recommended_action="Refer to Special Investigations Unit; hold payment.",
                evidence_links=["fraud_score.json:siu_referral"],
                assigned_to="SIU",
            )
        elif fraud.get("adjuster_review") is True:
            add(
                category="fraud_review",
                severity=Severity.medium,
                description="Elevated fraud indicators below the SIU threshold.",
                recommended_action="Adjuster to review fraud indicators before settlement.",
                evidence_links=["fraud_score.json:adjuster_review"],
                assigned_to="adjuster",
            )

        if coverage and coverage.get("coverage_active") is False:
            add(
                category="coverage_denial",
                severity=Severity.high,
                description=(
                    "Coverage not active: "
                    f"{coverage.get('denial_reason') or 'see coverage result'}."
                ),
                recommended_action="Issue denial notice; no payment.",
                evidence_links=["coverage_result.json:coverage_active"],
                assigned_to="adjuster",
            )

        if claim_summary.get("requires_manual_review") is True:
            add(
                category="data_quality",
                severity=Severity.high,
                description="Extraction confidence below auto-processing threshold.",
                recommended_action="Re-capture or manually verify low-confidence fields.",
                evidence_links=["claim_summary.json:requires_manual_review"],
                assigned_to="adjuster",
            )

        if settlement.get("total_loss_flag") is True:
            add(
                category="total_loss",
                severity=Severity.high,
                description="Repair-to-market ratio indicates a total loss.",
                recommended_action="Initiate total-loss / market-value valuation.",
                evidence_links=["settlement_calc.json:total_loss_flag"],
                assigned_to="adjuster",
            )

        if coverage.get("late_notice_violation") is True:
            add(
                category="late_notice",
                severity=Severity.medium,
                description=(
                    "Claim reported after the policy notice window "
                    f"({coverage.get('days_to_report')} of "
                    f"{coverage.get('late_notice_days_allowed')} days)."
                ),
                recommended_action="Review late-notice impact on coverage.",
                evidence_links=["coverage_result.json:late_notice_violation"],
                assigned_to="adjuster",
            )

        if flags.get("duplicate_claim") is True:
            add(
                category="duplicate_claim",
                severity=Severity.high,
                description="Claim flagged as a possible duplicate submission.",
                recommended_action="Verify against prior claims before payment.",
                evidence_links=["context_packet.json:flags.duplicate_claim"],
                assigned_to="adjuster",
            )

        # Invoice / PO / GRN matching exception (REQ-043).
        if settlement.get("invoice_match_status") == "mismatch":
            add(
                category="billing_variance",
                severity=Severity.medium,
                description=(
                    "Invoice/PO/GRN mismatch — billing variance "
                    f"{settlement.get('billing_variance_pct')}."
                ),
                recommended_action="Review invoice against PO/GRN before settlement.",
                evidence_links=["settlement_calc.json:invoice_match_status"],
                assigned_to="adjuster",
            )

        # NOTE: Agent D's repair/medical variance findings are not turned into a
        # separate exception here — the authoritative billing-variance exception
        # (above) already comes from the invoice/PO/GRN matcher, so surfacing
        # both would double-flag the same underlying issue.

        # Senior review for large gross amounts.
        gross = self._as_float(settlement.get("gross_amount"))
        senior_threshold = float(self.approval.get("senior_review_threshold", 50000.0))
        if settlement and gross > senior_threshold:
            add(
                category="senior_review",
                severity=Severity.medium,
                description=(
                    f"Gross amount {gross:,.2f} exceeds senior-review threshold."
                ),
                recommended_action="Escalate to a senior adjuster for authorisation.",
                evidence_links=["settlement_calc.json:gross_amount"],
                assigned_to="senior_adjuster",
            )

        # Regulatory-compliance failure (Agent G / REQ-048).
        if compliance and compliance.get("compliant") is False:
            failed = [
                c.get("check")
                for c in (compliance.get("checks") or [])
                if c.get("passed") is False
            ]
            add(
                category="compliance",
                severity=Severity.high,
                description=(
                    f"Failed {compliance.get('framework', 'NAIC')} compliance "
                    f"checks: {', '.join(str(f) for f in failed) or 'see report'}."
                ),
                recommended_action="Resolve the compliance gap before settlement.",
                evidence_links=["compliance_result.json:compliant"],
                assigned_to="compliance",
            )

        # Statistical anomaly worth a look (Agent / REQ-052). Billing-variance and
        # CAT-surge anomalies are excluded — they are already covered above.
        if anomaly:
            flagged = {
                a.get("code")
                for a in (anomaly.get("anomalies") or [])
                if a.get("code") in ("HIGH_VALUE_OUTLIER", "REPAIR_EXCEEDS_VALUE")
            }
            if flagged:
                add(
                    category="anomaly_review",
                    severity=Severity.medium,
                    description=(
                        "Statistical anomaly detected: "
                        f"{', '.join(sorted(str(c) for c in flagged))} "
                        f"(score {anomaly.get('anomaly_score')})."
                    ),
                    recommended_action="Investigate the outlier before settlement.",
                    evidence_links=["anomaly_report.json:anomalies"],
                    assigned_to="adjuster",
                )

        return exceptions

    # ------------------------------------------------------------------
    # Evidence + metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _evidence_bundle(findings: list[Finding], run_path: Path) -> list[str]:
        bundle: set[str] = set()
        for finding in findings:
            bundle.update(finding.evidence_links)
        for name in _UPSTREAM_ARTIFACTS:
            if (run_path / name).is_file():
                bundle.add(name)
        return sorted(bundle)

    def _build_metrics(
        self,
        context: dict,
        claim_summary: dict,
        coverage: dict,
        settlement: dict,
        fraud: dict,
        findings: list[Finding],
        exceptions: list[ExceptionEntry],
        routing: RoutingDecision,
        pipeline_duration_seconds: float,
        agents_executed: list[str] | None,
        cache_hit: bool,
        liability: dict | None = None,
        compliance: dict | None = None,
        anomaly: dict | None = None,
    ) -> RunMetrics:
        if agents_executed is None:
            agents_executed = ["agent_fnol_intake"]
            if claim_summary:
                agents_executed.append("agent_document_extraction")
            if coverage:
                agents_executed.append("agent_coverage_validation")
            if settlement:
                agents_executed.append("agent_damage_assessment")
            if fraud:
                agents_executed.append("agent_fraud_detection")
            if liability:
                agents_executed.append("agent_liability")
            if compliance:
                agents_executed.append("agent_compliance")
            if anomaly:
                agents_executed.append("agent_anomaly")
            agents_executed.append("agent_exception_triage")

        by_severity: dict[str, int] = {}
        for finding in findings:
            key = finding.severity.value
            by_severity[key] = by_severity.get(key, 0) + 1

        return RunMetrics(
            claim_id=context["claim_id"],
            run_id=str(context.get("run_id", context["claim_id"])),
            pipeline_duration_seconds=round(float(pipeline_duration_seconds), 4),
            agents_executed=agents_executed,
            total_findings=len(findings),
            findings_by_severity=by_severity,
            extraction_field_count=len(claim_summary.get("extracted_fields", []) or []),
            extraction_accuracy=round(
                self._as_float(claim_summary.get("overall_confidence")), 4
            ),
            low_confidence_field_count=len(
                claim_summary.get("low_confidence_fields", []) or []
            ),
            exception_count=len(exceptions),
            # Share of ACTIONABLE findings (>= medium severity) that became
            # exceptions (REQ-029). Info/low findings are excluded so advisory
            # notes do not dilute the rate. Capped at 1.0.
            exception_rate=self._exception_rate(findings, len(exceptions)),
            routing_decision=routing.value,
            input_hash=str(context.get("input_hash", "")),
            cache_hit=bool(cache_hit),
        )

    # ------------------------------------------------------------------
    # Markdown writers
    # ------------------------------------------------------------------

    def _redact_pii(self, text: str) -> str:
        """
        Apply policy-driven PII redaction to human-readable artifacts
        (REQ-003). Driven entirely by config/policy.yaml privacy flags so the
        compliance posture is reconfigurable without code changes.
        """
        if not self.privacy.get("audit_log_pii_redaction", False):
            return text
        if self.privacy.get("mask_ssn", False):
            text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "***-**-****", text)
        if self.privacy.get("mask_dob", False):
            # Mask a YYYY-MM-DD date only when it is labelled a date of birth,
            # so settlement/finalisation dates remain intact.
            text = re.sub(
                r"(?i)(date of birth|dob)(\s*[:=]?\s*)\d{4}-\d{2}-\d{2}",
                r"\1\2****-**-**",
                text,
            )
        return text

    def _write_exceptions_md(
        self,
        path: Path,
        claim_id: str,
        routing: RoutingDecision,
        net_settlement: float,
        exceptions: list[ExceptionEntry],
    ) -> None:
        lines = [
            f"# Exception Summary: {claim_id}",
            "",
            f"- Routing decision: **{routing.value}**",
            f"- Net settlement: **${net_settlement:,.2f}**",
            f"- Exceptions raised: **{len(exceptions)}**",
            "",
        ]
        if not exceptions:
            lines.append("No exceptions. Claim cleared for straight-through processing.")
        else:
            for exc in exceptions:
                assignee = exc.assigned_to or "unassigned"
                lines.extend(
                    [
                        f"## {exc.exception_id} — {exc.category} ({exc.severity.value})",
                        f"- Description: {exc.description}",
                        f"- Recommended action: {exc.recommended_action}",
                        f"- Assigned to: {assignee}",
                        f"- Evidence: {', '.join(exc.evidence_links)}",
                        "",
                    ]
                )
        path.write_text(
            self._redact_pii("\n".join(lines).rstrip() + "\n"), encoding="utf-8"
        )

    def _write_audit_log(
        self,
        path: Path,
        claim_id: str,
        run_id: str,
        input_hash: str,
        finalized_at: str,
        routing: RoutingDecision,
        net_settlement: float,
        rationale: str,
        findings: list[Finding],
        exceptions: list[ExceptionEntry],
        evidence_bundle: list[str],
    ) -> None:
        lines = [
            f"# SICPS Audit Log: {claim_id}",
            "",
            f"- Run ID: {run_id}",
            f"- Input hash: {input_hash}",
            f"- Finalized at: {finalized_at}",
            "",
            "## Routing Decision",
            f"- Decision: **{routing.value}**",
            f"- Net settlement: **${net_settlement:,.2f}**",
            f"- Rationale: {rationale}",
            "",
            "## Findings (merged, deduplicated, prioritised)",
        ]

        if not findings:
            lines.append("- No findings recorded.")
        else:
            current_agent = None
            for finding in findings:
                if finding.agent != current_agent:
                    current_agent = finding.agent
                    lines.append(f"### {current_agent}")
                evidence = ", ".join(finding.evidence_links) or "n/a"
                lines.append(
                    f"- [{finding.severity.value}] {finding.finding_id}: "
                    f"{finding.recommendation} (evidence: {evidence})"
                )

        lines.extend(["", "## Exceptions"])
        if not exceptions:
            lines.append("- None.")
        else:
            for exc in exceptions:
                lines.append(
                    f"- [{exc.severity.value}] {exc.exception_id} "
                    f"({exc.category}) -> {exc.recommended_action}"
                )

        lines.extend(["", "## Evidence Bundle"])
        lines.extend(f"- {item}" for item in evidence_bundle)

        path.write_text(
            self._redact_pii("\n".join(lines).rstrip() + "\n"), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _exception_rate(findings: list[Finding], exception_count: int) -> float:
        """Exceptions as a share of actionable (>= medium) findings, capped at 1.0."""
        actionable = sum(
            1
            for f in findings
            if f.severity in (Severity.critical, Severity.high, Severity.medium)
        )
        if actionable <= 0:
            return 0.0
        return round(min(exception_count / actionable, 1.0), 4)

    @staticmethod
    def _as_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    @staticmethod
    def _require_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(
                f"context_packet.json not found; Agent A must run first: {path}"
            )
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _optional_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _write_settlement_csv(
        path: Path,
        claim_id: str,
        run_id: str,
        routing: RoutingDecision,
        net_settlement: float,
        settlement: dict,
        coverage: dict,
        fraud: dict,
        exception_count: int,
        finalized_at: str,
        input_hash: str,
    ) -> None:
        """
        CoreLogic-ready settlement posting payload as a single-row CSV
        (REQ-004). Fixed column order + deterministic values keep it
        byte-for-byte reproducible (REQ-044).
        """
        columns = [
            "claim_id",
            "run_id",
            "routing_decision",
            "net_settlement",
            "gross_amount",
            "deductible",
            "coverage_active",
            "total_loss_flag",
            "fraud_score",
            "risk_level",
            "invoice_match_status",
            "billing_variance_pct",
            "exception_count",
            "finalized_at",
            "input_hash",
        ]
        row = {
            "claim_id": claim_id,
            "run_id": run_id,
            "routing_decision": routing.value,
            "net_settlement": net_settlement,
            "gross_amount": AgentExceptionTriage._as_float(settlement.get("gross_amount")),
            "deductible": AgentExceptionTriage._as_float(settlement.get("deductible")),
            "coverage_active": coverage.get("coverage_active", ""),
            "total_loss_flag": settlement.get("total_loss_flag", ""),
            "fraud_score": AgentExceptionTriage._as_float(fraud.get("fraud_score")),
            "risk_level": fraud.get("risk_level", ""),
            "invoice_match_status": settlement.get("invoice_match_status", "not_evaluated"),
            "billing_variance_pct": AgentExceptionTriage._as_float(
                settlement.get("billing_variance_pct")
            ),
            "exception_count": exception_count,
            "finalized_at": finalized_at,
            "input_hash": input_hash,
        }
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _write_settlement_payload_json(
        path: Path,
        claim_id: str,
        run_id: str,
        routing: RoutingDecision,
        net_settlement: float,
        settlement: dict,
        coverage: dict,
        fraud: dict,
        exception_count: int,
        finalized_at: str,
        input_hash: str,
    ) -> None:
        """
        CoreLogic-ready settlement posting payload as structured JSON
        (optional artifact #9). Deterministic key order keeps it byte-for-byte
        reproducible (REQ-044).
        """
        f = AgentExceptionTriage._as_float
        payload = {
            "claim_id": claim_id,
            "run_id": run_id,
            "routing_decision": routing.value,
            "settlement": {
                "net_settlement": net_settlement,
                "gross_amount": f(settlement.get("gross_amount")),
                "deductible": f(settlement.get("deductible")),
                "total_loss_flag": bool(settlement.get("total_loss_flag", False)),
            },
            "coverage": {
                "coverage_active": coverage.get("coverage_active"),
                "denial_reason": coverage.get("denial_reason"),
            },
            "fraud": {
                "fraud_score": f(fraud.get("fraud_score")),
                "risk_level": fraud.get("risk_level", ""),
                "siu_referral": bool(fraud.get("siu_referral", False)),
            },
            "matching": {
                "invoice_match_status": settlement.get(
                    "invoice_match_status", "not_evaluated"
                ),
                "billing_variance_pct": f(settlement.get("billing_variance_pct")),
            },
            "exception_count": exception_count,
            "finalized_at": finalized_at,
            "input_hash": input_hash,
        }
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")

    # Routing → tracker assignee / priority maps (deterministic).
    _TRACKER_ASSIGNEE = {
        "siu_referral": "SIU",
        "coverage_denial": "adjuster",
        "senior_review": "senior_adjuster",
        "total_loss_routing": "adjuster",
        "late_notice_review": "adjuster",
        "manual_review_route": "adjuster",
        "adjuster_review": "adjuster",
        "cat_surge_processing": "auto",
        "auto_settle": "auto",
    }
    _TRACKER_PRIORITY = {
        "siu_referral": "critical",
        "coverage_denial": "high",
        "senior_review": "high",
        "total_loss_routing": "high",
        "late_notice_review": "medium",
        "manual_review_route": "medium",
        "adjuster_review": "medium",
        "cat_surge_processing": "low",
        "auto_settle": "low",
    }

    @classmethod
    def _write_tracker_ticket(
        cls,
        path: Path,
        claim_id: str,
        run_id: str,
        routing: RoutingDecision,
        net_settlement: float,
        exceptions: list[ExceptionEntry],
        adjuster_notes: str,
        finalized_at: str,
        input_hash: str,
    ) -> None:
        """
        REQ-051 stub — the work item we would POST to ADO/Jira. Deterministic
        and idempotent: the ticket id is derived from the claim id, so the same
        claim always maps to the same ticket (no duplicates on re-run).
        Auto-settle / CAT surge are auto-closed; everything else is open work.
        """
        routing_value = routing.value
        auto = routing_value in ("auto_settle", "cat_surge_processing")
        ticket = {
            "ticket_id": f"SICPS-{claim_id}",
            "tracker": "ADO/Jira (stub)",
            "claim_id": claim_id,
            "run_id": run_id,
            "title": f"[{routing_value}] Claim {claim_id}",
            "status": "closed" if auto else "open",
            "assigned_to": cls._TRACKER_ASSIGNEE.get(routing_value, "adjuster"),
            "priority": cls._TRACKER_PRIORITY.get(routing_value, "medium"),
            "routing_decision": routing_value,
            "net_settlement": net_settlement,
            "exception_count": len(exceptions),
            "exceptions": [
                {"id": e.exception_id, "category": e.category, "action": e.recommended_action}
                for e in exceptions
            ],
            "summary": adjuster_notes,
            "created_at": finalized_at,
            "input_hash": input_hash,
        }
        with path.open("w", encoding="utf-8") as file:
            json.dump(ticket, file, indent=2, sort_keys=True)
            file.write("\n")

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
            file.write("\n")
