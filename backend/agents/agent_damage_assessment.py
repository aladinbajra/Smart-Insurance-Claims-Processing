import json

from pathlib import Path
from typing import Any

from backend.models.schemas import (
    SettlementCalc,
    LineItem,
    Finding,
    Severity,
)
from backend.tools.pdf_extractor import load_policy
from backend.tools.invoice_matcher import match_invoice


class AgentDamageAssessment:
    def __init__(
        self,
        config_path: str | Path,
        runs_dir: str | Path,
    ) -> None:

        self.config_path = Path(config_path)
        self.runs_dir = Path(runs_dir)
        self.policy_pack = load_policy(self.config_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_context(self, claim_id: str) -> dict[str, Any]:
        """
        Load Agent A's context packet (run timestamp + invoice matching inputs).
        Using the canonical run timestamp keeps settlement_calc.json
        byte-for-byte reproducible (REQ-044) instead of wall-clock now().
        """
        context_path = self.runs_dir / claim_id / "context_packet.json"
        if context_path.is_file():
            with context_path.open("r", encoding="utf-8") as file:
                return json.load(file)
        return {}

    @staticmethod
    def _create_finding(
        finding_id: str,
        severity: Severity,
        recommendation: str,
        evidence_links: list[str],
        timestamp: str,
        confidence: float = 1.0,
        requires_human_review: bool = True,
    ) -> Finding:
        return Finding(
            finding_id=finding_id,
            agent="agent_damage_assessment",
            severity=severity,
            confidence=confidence,
            evidence_links=evidence_links,
            recommendation=recommendation,
            open_questions=[],
            requires_human_review=requires_human_review,
            timestamp=timestamp,
        )

    @staticmethod
    def _write_json(
        path: Path,
        data: dict[str, Any],
    ) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
            file.write("\n")

    @staticmethod
    def _write_audit_log(
        path: Path,
        claim_id: str,
        entries: list[str],
    ) -> None:
        lines = [
            f"# Settlement Audit: {claim_id}",
            "",
        ]
        lines.extend(f"- {entry}" for entry in entries)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _calculate_repair_variance(
        repair_estimate: float,
        gross_amount: float,
    ) -> float:
        """
        Fix D-4: Compute repair estimate as a share of total gross amount.
        This is bounded [0, 1] and meaningful as a composition check.
        Compare against tolerances.repair_cost_variance_pct (0.25).
        Original divided by market_value, producing ratios that falsely
        exceeded the 25% tolerance on almost any significant repair.
        """
        if gross_amount <= 0:
            return 0.0
        return round(repair_estimate / gross_amount, 4)

    @staticmethod
    def _calculate_medical_variance(
        medical_total: float,
        gross_amount: float,
    ) -> float:
        """
        Fix D-5: Medical costs as a share of gross settlement.
        Retained from original — no external medical benchmark exists
        in PolicyPack or FinancialInfo schemas. Bounded [0, 1].
        Compare against tolerances.medical_cost_variance_pct (0.30).
        """
        if gross_amount <= 0:
            return 0.0
        return round(medical_total / gross_amount, 4)

    @staticmethod
    def _is_within_tolerance(
        value: float,
        tolerance: float,
    ) -> bool:
        return value <= tolerance

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(
        self,
        claim_summary: dict,
        coverage_result: dict,
    ) -> SettlementCalc:
        """
        Parameters
        ----------
        claim_summary:
            Parsed content of runs/<claim_id>/claim_summary.json
            produced by Agent B. REQ-038.
        coverage_result:
            Parsed content of runs/<claim_id>/coverage_result.json
            produced by Agent C. REQ-039.
        """

        claim_id: str = claim_summary["claim_id"]

        # Load Agent A context once: deterministic timestamp (REQ-044) +
        # invoice/PO/GRN matching inputs (REQ-043).
        context = self._load_context(claim_id)
        created_at = str(context.get("created_at", "1970-01-01T00:00:00Z"))
        invoice_matching = context.get("invoice_matching") or {}

        findings: list[Finding] = []
        audit_entries: list[str] = []

        # --------------------------------------------------------------
        # OCR / missing-document guard — derived from Agent B output
        # --------------------------------------------------------------
        overall_confidence = claim_summary.get("overall_confidence")

        missing_documents: bool = bool(claim_summary.get("extraction_errors"))

        ocr_min: float = self.policy_pack.extraction.min_confidence_auto_process

        low_ocr_flag: bool = (
            overall_confidence is not None and overall_confidence < ocr_min
        )

        unresolvable = low_ocr_flag or missing_documents

        if unresolvable:
            error_descriptor = (
                "missing_documents" if missing_documents else "low_ocr_confidence"
            )

            audit_entries.append(f"Processing halted: {error_descriptor}")

            findings.append(
                self._create_finding(
                    finding_id="D-007-UNRESOLVABLE_FILE",
                    severity=Severity.high,
                    recommendation=(
                        f"Manual review required: {error_descriptor}. "
                        "Settlement cannot be computed automatically."
                    ),
                    evidence_links=[
                        "claim_summary.overall_confidence",
                        "claim_summary.extraction_errors",
                    ],
                    timestamp=created_at,
                )
            )

            audit_entries.append(f"Settlement findings generated: {len(findings)}")
            audit_entries.append("Settlement processing completed.")

            result = SettlementCalc(
                claim_id=claim_id,
                gross_amount=0.0,
                deductible=coverage_result.get("deductible", 0.0),
                net_settlement=0.0,
                line_items=[],
                repair_estimate=0.0,
                medical_total=0.0,
                replacement_value=0.0,
                market_value=0.0,
                total_loss_ratio=0.0,
                total_loss_flag=False,
                repair_variance_pct=0.0,
                medical_variance_pct=0.0,
                repair_within_tolerance=True,
                medical_within_tolerance=True,
                billing_variance_pct=0.0,
                invoice_match_status="not_evaluated",
                findings=findings,
                finalized=False,
            )

            self._persist(claim_id, result, audit_entries)
            return result

        # --------------------------------------------------------------
        # Extract financials — from claim_summary (Agent B) and
        # coverage_result (Agent C). REQ-038–039.
        # --------------------------------------------------------------
        repair_estimate: float = claim_summary["repair_estimate"]
        medical_total: float = claim_summary["medical_total"]
        market_value: float = claim_summary["market_value"]

        # Deductible comes from Agent C (policy-validated value)
        deductible: float = coverage_result["deductible"]

        # ClaimSummary carries no separate injury_claim line.
        injury_claim: float = 0.0

        # --------------------------------------------------------------
        # D-006: Missing market value
        # --------------------------------------------------------------
        if market_value <= 0:
            findings.append(
                self._create_finding(
                    finding_id="D-006-MISSING_MARKET_VALUE",
                    severity=Severity.low,
                    recommendation=(
                        "Market value unavailable. " "Variance calculations limited."
                    ),
                    evidence_links=[
                        "claim_summary.market_value",
                    ],
                    timestamp=created_at,
                )
            )
            audit_entries.append("Market value missing.")

        # --------------------------------------------------------------
        # Total loss detection (computed before settlement composition)
        # ratio = repair_estimate / market_value; threshold from policy_pack
        # --------------------------------------------------------------
        total_loss_ratio: float = (
            repair_estimate / market_value if market_value > 0 else 0.0
        )

        total_loss_threshold: float = (
            self.policy_pack.approval.total_loss_ratio_threshold
        )

        total_loss_flag: bool = total_loss_ratio >= total_loss_threshold

        # --------------------------------------------------------------
        # Settlement composition (REQ-019: repair, medical, replacement)
        # A total loss pays the market / replacement value, not the repair
        # cost — you do not pay repair on a vehicle worth less than repair.
        # --------------------------------------------------------------
        if total_loss_flag and market_value > 0:
            replacement_value: float = market_value
            vehicle_component: float = market_value
            vehicle_line = LineItem(
                description="Replacement Value (Total Loss)",
                amount=market_value,
                category="replacement",
            )
        else:
            replacement_value = 0.0
            vehicle_component = repair_estimate
            vehicle_line = LineItem(
                description="Repair Estimate",
                amount=repair_estimate,
                category="repair",
            )

        line_items: list[LineItem] = [
            vehicle_line,
            LineItem(
                description="Medical Costs",
                amount=medical_total,
                category="medical",
            ),
        ]

        if injury_claim > 0:
            line_items.append(
                LineItem(
                    description="Injury Claim",
                    amount=injury_claim,
                    category="injury",
                )
            )

        # --------------------------------------------------------------
        # Gross amount = vehicle component (repair or replacement) + medical
        # --------------------------------------------------------------
        gross_amount: float = vehicle_component + medical_total + injury_claim

        audit_entries.append(f"Gross amount calculated (Agent D): {gross_amount}")

        # --------------------------------------------------------------
        # Net settlement
        # --------------------------------------------------------------
        net_settlement: float = max(gross_amount - deductible, 0.0)

        audit_entries.append(f"Net settlement calculated: {net_settlement}")

        if total_loss_flag:
            findings.append(
                self._create_finding(
                    finding_id="D-001-TOTAL_LOSS",
                    severity=Severity.high,
                    recommendation="Escalate total loss settlement.",
                    evidence_links=[
                        "claim_summary.repair_estimate",
                        "claim_summary.market_value",
                    ],
                    timestamp=created_at,
                )
            )
            audit_entries.append("Total loss detected.")

        # --------------------------------------------------------------
        # D-002: Gross wiped by deductible
        # --------------------------------------------------------------
        if gross_amount <= deductible:
            findings.append(
                self._create_finding(
                    finding_id="D-002-NEGATIVE_SETTLEMENT",
                    severity=Severity.medium,
                    recommendation="Review deductible application.",
                    evidence_links=[
                        "coverage_result.deductible",
                    ],
                    timestamp=created_at,
                )
            )
            audit_entries.append("Settlement reduced to zero.")

        # --------------------------------------------------------------
        # Variance calculations
        # Fix D-4: repair_estimate / gross_amount (bounded, meaningful)
        # Fix D-5: medical_total / gross_amount (retained, bounded)
        # --------------------------------------------------------------
        repair_variance_pct: float = self._calculate_repair_variance(
            repair_estimate,
            gross_amount,
        )

        medical_variance_pct: float = self._calculate_medical_variance(
            medical_total,
            gross_amount,
        )

        repair_within_tolerance: bool = self._is_within_tolerance(
            repair_variance_pct,
            self.policy_pack.tolerances.repair_cost_variance_pct,
        )

        medical_within_tolerance: bool = self._is_within_tolerance(
            medical_variance_pct,
            self.policy_pack.tolerances.medical_cost_variance_pct,
        )

        # --------------------------------------------------------------
        # D-003: Repair variance exceeded
        # --------------------------------------------------------------
        if not repair_within_tolerance:
            findings.append(
                self._create_finding(
                    finding_id="D-003-REPAIR_VARIANCE_EXCEEDED",
                    severity=Severity.medium,
                    recommendation="Review repair estimate variance.",
                    evidence_links=[
                        "claim_summary.repair_estimate",
                    ],
                    timestamp=created_at,
                )
            )
            audit_entries.append("Repair variance exceeded tolerance.")

        # --------------------------------------------------------------
        # D-004: Medical variance exceeded
        # --------------------------------------------------------------
        if not medical_within_tolerance:
            findings.append(
                self._create_finding(
                    finding_id="D-004-MEDICAL_VARIANCE_EXCEEDED",
                    severity=Severity.medium,
                    recommendation="Review medical estimate variance.",
                    evidence_links=[
                        "claim_summary.medical_total",
                    ],
                    timestamp=created_at,
                )
            )
            audit_entries.append("Medical variance exceeded tolerance.")

        # --------------------------------------------------------------
        # D-005: Replacement exceeds market value
        # --------------------------------------------------------------
        if replacement_value > market_value and market_value > 0:
            findings.append(
                self._create_finding(
                    finding_id="D-005-REPLACEMENT_EXCEEDS_MARKET",
                    severity=Severity.medium,
                    recommendation="Review replacement value against market value.",
                    evidence_links=[
                        "claim_summary.market_value",
                    ],
                    timestamp=created_at,
                )
            )
            audit_entries.append("Replacement value exceeds market value.")

        # --------------------------------------------------------------
        # Invoice ↔ PO ↔ GRN matching (REQ-043: Matching Precision)
        # --------------------------------------------------------------
        billing_variance_pct = 0.0
        invoice_match_status = "not_evaluated"

        if invoice_matching:
            match = match_invoice(
                invoice_amount=float(invoice_matching.get("invoice_amount", 0.0)),
                expected_amount=float(invoice_matching.get("expected_amount", 0.0)),
                line_variance_pct=float(invoice_matching.get("variance_pct", 0.0)),
                po_reference=invoice_matching.get("po_reference"),
                grn_reference=invoice_matching.get("grn_reference"),
                tolerance=self.policy_pack.tolerances.repair_cost_variance_pct,
            )
            billing_variance_pct = match.billing_variance_pct
            invoice_match_status = match.match_status

            audit_entries.append(
                f"Invoice match: {invoice_match_status} "
                f"(billing variance {billing_variance_pct})."
            )

            if match.match_status == "mismatch":
                findings.append(
                    self._create_finding(
                        finding_id="D-008-INVOICE_MISMATCH",
                        severity=Severity.medium,
                        recommendation=(
                            "Invoice does not match PO/expected amount: "
                            f"{'; '.join(match.reasons)}. Review billing variance."
                        ),
                        evidence_links=[
                            "context_packet.invoice_matching",
                            "settlement_calc.billing_variance_pct",
                        ],
                        timestamp=created_at,
                    )
                )

        # --------------------------------------------------------------
        # Finalize audit trail
        # --------------------------------------------------------------
        audit_entries.append(f"Settlement findings generated: {len(findings)}")
        audit_entries.append("Settlement processing completed.")

        result = SettlementCalc(
            claim_id=claim_id,
            gross_amount=gross_amount,
            deductible=deductible,
            net_settlement=net_settlement,
            line_items=line_items,
            repair_estimate=repair_estimate,
            medical_total=medical_total,
            replacement_value=replacement_value,
            market_value=market_value,
            total_loss_ratio=round(total_loss_ratio, 4),
            total_loss_flag=total_loss_flag,
            repair_variance_pct=repair_variance_pct,
            medical_variance_pct=medical_variance_pct,
            repair_within_tolerance=repair_within_tolerance,
            medical_within_tolerance=medical_within_tolerance,
            billing_variance_pct=billing_variance_pct,
            invoice_match_status=invoice_match_status,
            findings=findings,
            finalized=False,
        )

        self._persist(claim_id, result, audit_entries)
        return result

    # ------------------------------------------------------------------
    # Persistence — extracted so both code paths use the same logic
    # ------------------------------------------------------------------

    def _persist(
        self,
        claim_id: str,
        result: SettlementCalc,
        audit_entries: list[str],
    ) -> None:
        run_dir = self.runs_dir / claim_id
        run_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(
            run_dir / "settlement_calc.json",
            result.model_dump(mode="json"),
        )

        self._write_audit_log(
            run_dir / "settlement_audit.md",
            claim_id,
            audit_entries,
        )
