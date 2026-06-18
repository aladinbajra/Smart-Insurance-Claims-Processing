import json

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from backend.models.schemas import (
    CoverageResult,
    Finding,
    Severity,
)


class AgentCoverageValidation:
    """
    Agent C: Coverage & Policy Validation.

    Responsibilities:
    - Verify policy status dynamically from dates.
    - Detect expired policies.
    - Detect pre-inception loss events.
    - Evaluate policy exclusions against claim details.
    - Detect late notice violations.
    - Compute waiting period / lookback window.
    - Produce coverage findings.
    - Persist coverage_result.json to runs/<claim_id>/.
    """

    def __init__(
        self,
        runs_dir: str | Path,
        config_path: str | Path | None = None,
    ) -> None:

        self.runs_dir = Path(runs_dir)

        # Deterministic run timestamp; set per-claim in process().
        self._created_at: str = "1970-01-01T00:00:00Z"

        # Regional state overrides (REQ-037) and coverage sub-limits (REQ-017),
        # loaded from policy.yaml when a config path is supplied.
        self._state_overrides: dict[str, int] = {}
        self._sub_limits: dict[str, float] = {}
        if config_path is not None:
            cfg = self._load_yaml(Path(config_path))
            overrides = (cfg.get("compliance", {}) or {}).get("state_overrides", {}) or {}
            self._state_overrides = {
                state: int(rule["late_notice_days"])
                for state, rule in overrides.items()
                if isinstance(rule, dict) and rule.get("late_notice_days") is not None
            }
            self._sub_limits = (cfg.get("coverage", {}) or {}).get("sub_limits", {}) or {}

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    @staticmethod
    def _write_json(
        path: Path,
        data: dict[str, Any],
    ) -> None:

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                sort_keys=True,
            )

            file.write("\n")

    def _load_created_at(self, claim_id: str) -> str:
        """
        Deterministic run timestamp taken from Agent A's context packet.
        Using the canonical run timestamp (instead of wall-clock now())
        keeps coverage_result.json byte-for-byte reproducible (REQ-044).
        """
        context_path = self.runs_dir / claim_id / "context_packet.json"
        if context_path.is_file():
            with context_path.open("r", encoding="utf-8") as file:
                return str(
                    json.load(file).get("created_at", "1970-01-01T00:00:00Z")
                )
        return "1970-01-01T00:00:00Z"

    @staticmethod
    def _parse_date(value: str) -> datetime:

        return datetime.fromisoformat(value).replace(
            tzinfo=timezone.utc
        )

    def _make_finding(
        self,
        finding_id: str,
        severity: Severity,
        recommendation: str,
        evidence_links: list[str],
    ) -> Finding:

        return Finding(
            finding_id=finding_id,
            agent="agent_coverage_validation",
            severity=severity,
            confidence=1.0,
            evidence_links=evidence_links,
            recommendation=recommendation,
            open_questions=[],
            requires_human_review=True,
            timestamp=self._created_at,
        )

    def process(
        self,
        claim_summary: dict,
        policy: dict,
    ) -> CoverageResult:
        """
        Parameters
        ----------
        claim_summary:
            Parsed content of runs/<claim_id>/claim_summary.json
            produced by Agent B.
        policy:
            Parsed content of policy.yaml for this claim.
        """

        findings: list[Finding] = []

        claim_id: str = claim_summary["claim_id"]

        # Deterministic timestamp for all findings (REQ-044).
        self._created_at = self._load_created_at(claim_id)

        coverage_active: bool = True
        denial_reason: str | None = None
        exclusions_triggered: list[str] = []
        waiting_period_active: bool = False

        # ----------------------------------------------------------
        # Derived dates — incident_date is denormalized directly on the
        # claim summary (Agent B output), not nested under an "incident" object.
        # ----------------------------------------------------------
        incident_date = self._parse_date(claim_summary["incident_date"])
        effective_date = self._parse_date(policy["effective_date"])
        expiration_date = self._parse_date(policy["expiration_date"])

        # ----------------------------------------------------------
        # Rule 1: Policy Expired — computed dynamically from dates
        # ----------------------------------------------------------
        policy_expired: bool = incident_date > expiration_date

        if policy_expired:
            coverage_active = False
            denial_reason = "Policy expired"

            findings.append(
                self._make_finding(
                    finding_id="C-001-POLICY_EXPIRED",
                    severity=Severity.high,
                    recommendation=(
                        "Deny claim and escalate for coverage review."
                    ),
                    evidence_links=[
                        "policy.expiration_date",
                        "claim_summary.incident_date",
                    ],
                )
            )

            # Terminal denial — skip all downstream checks
            return self._finalize(
                claim_id=claim_id,
                claim_summary=claim_summary,
                policy=policy,
                coverage_active=coverage_active,
                policy_expired=policy_expired,
                denial_reason=denial_reason,
                exclusions_triggered=exclusions_triggered,
                waiting_period_active=waiting_period_active,
                findings=findings,
            )

        # ----------------------------------------------------------
        # Rule 2: Pre-inception loss event — computed from dates
        # ----------------------------------------------------------
        pre_inception: bool = incident_date < effective_date

        if pre_inception:
            coverage_active = False
            denial_reason = "Pre-inception loss event"

            findings.append(
                self._make_finding(
                    finding_id="C-002-COVERAGE_DENIED",
                    severity=Severity.high,
                    recommendation=(
                        "Coverage does not apply: loss predates policy effective date."
                    ),
                    evidence_links=[
                        "policy.effective_date",
                        "claim_summary.incident_date",
                    ],
                )
            )

            # Terminal denial — skip all downstream checks
            return self._finalize(
                claim_id=claim_id,
                claim_summary=claim_summary,
                policy=policy,
                coverage_active=coverage_active,
                policy_expired=policy_expired,
                denial_reason=denial_reason,
                exclusions_triggered=exclusions_triggered,
                waiting_period_active=waiting_period_active,
                findings=findings,
            )

        # ----------------------------------------------------------
        # Rule 3: Policy Exclusions — iterate against claim details
        # ----------------------------------------------------------
        policy_exclusions: list[str] = policy.get("exclusions", [])

        # Match exclusions against the text Agent B actually extracted from the
        # documents (claim_summary.extracted_fields), not manifest ground truth.
        extracted_fields: list[dict] = claim_summary.get("extracted_fields", [])

        claim_text_fields: list[str] = [
            str(field.get("value", "")).lower()
            for field in extracted_fields
            if isinstance(field.get("value"), str)
        ]

        for exclusion in policy_exclusions:
            exclusion_lower = exclusion.lower()

            matched = any(
                exclusion_lower in field
                for field in claim_text_fields
            )

            if matched:
                exclusions_triggered.append(exclusion)

        if exclusions_triggered:
            coverage_active = False
            denial_reason = (
                f"Exclusion(s) triggered: {', '.join(exclusions_triggered)}"
            )

            findings.append(
                self._make_finding(
                    finding_id="C-002-COVERAGE_DENIED",
                    severity=Severity.high,
                    recommendation=(
                        "Coverage does not apply to this claim. "
                        "Review triggered exclusions with adjuster."
                    ),
                    evidence_links=[
                        "policy.exclusions",
                        "claim_summary.extracted_fields",
                    ],
                )
            )

            # Terminal denial — skip all downstream checks
            return self._finalize(
                claim_id=claim_id,
                claim_summary=claim_summary,
                policy=policy,
                coverage_active=coverage_active,
                policy_expired=policy_expired,
                denial_reason=denial_reason,
                exclusions_triggered=exclusions_triggered,
                waiting_period_active=waiting_period_active,
                findings=findings,
            )

        # ----------------------------------------------------------
        # Rule 4: Waiting Period / Lookback Window
        # ----------------------------------------------------------
        pre_existing_lookback_days: int = policy.get(
            "pre_existing_lookback_days",
            0,
        )

        if pre_existing_lookback_days > 0:
            days_since_effective = (
                incident_date - effective_date
            ).days

            waiting_period_active = (
                days_since_effective < pre_existing_lookback_days
            )

            if waiting_period_active:
                findings.append(
                    self._make_finding(
                        finding_id="C-004-WAITING_PERIOD_ACTIVE",
                        severity=Severity.medium,
                        recommendation=(
                            "Claim falls within the policy lookback window. "
                            "Escalate for pre-existing condition review."
                        ),
                        evidence_links=[
                            "policy.pre_existing_lookback_days",
                            "claim_summary.incident_date",
                            "policy.effective_date",
                        ],
                    )
                )

        # ----------------------------------------------------------
        # Rule 5: Late Notice — apply regional state override (REQ-037)
        # ----------------------------------------------------------
        days_to_report: int = claim_summary.get("days_to_report", 0)
        state: str = str(policy.get("state", ""))
        late_notice_days: int = self._state_overrides.get(
            state, policy.get("late_notice_days", 0)
        )

        late_notice_violation: bool = days_to_report > late_notice_days

        if late_notice_violation:
            findings.append(
                self._make_finding(
                    finding_id="C-003-LATE_NOTICE",
                    severity=Severity.medium,
                    recommendation=(
                        "Review late reporting before settlement."
                    ),
                    evidence_links=[
                        "claim_summary.days_to_report",
                        "policy.late_notice_days",
                    ],
                )
            )

        # ----------------------------------------------------------
        # Rule 6: Sub-limits — coverage caps per category (REQ-017)
        # ----------------------------------------------------------
        medical_total: float = float(claim_summary.get("medical_total", 0.0))
        medical_sub_limit: float = float(self._sub_limits.get("medical_max", 0.0))

        if medical_sub_limit > 0 and medical_total > medical_sub_limit:
            findings.append(
                self._make_finding(
                    finding_id="C-005-SUBLIMIT_EXCEEDED",
                    severity=Severity.medium,
                    recommendation=(
                        f"Medical total {medical_total} exceeds the policy "
                        f"sub-limit {medical_sub_limit}; cap the medical payout."
                    ),
                    evidence_links=[
                        "claim_summary.medical_total",
                        "policy.sub_limits.medical_max",
                    ],
                )
            )

        return self._finalize(
            claim_id=claim_id,
            claim_summary=claim_summary,
            policy=policy,
            coverage_active=coverage_active,
            policy_expired=policy_expired,
            denial_reason=denial_reason,
            exclusions_triggered=exclusions_triggered,
            waiting_period_active=waiting_period_active,
            findings=findings,
            late_notice_violation=late_notice_violation,
            days_to_report=days_to_report,
            late_notice_days_allowed=late_notice_days,
        )

    def _finalize(
        self,
        claim_id: str,
        claim_summary: dict,
        policy: dict,
        coverage_active: bool,
        policy_expired: bool,
        denial_reason: str | None,
        exclusions_triggered: list[str],
        waiting_period_active: bool,
        findings: list[Finding],
        late_notice_violation: bool = False,
        days_to_report: int = 0,
        late_notice_days_allowed: int | None = None,
    ) -> CoverageResult:

        if late_notice_days_allowed is None:
            late_notice_days_allowed = policy.get("late_notice_days", 0)

        result = CoverageResult(
            claim_id=claim_id,
            coverage_active=coverage_active,
            policy_expired=policy_expired,
            denial_reason=denial_reason,
            late_notice_violation=late_notice_violation,
            late_notice_days_allowed=late_notice_days_allowed,
            days_to_report=days_to_report or claim_summary.get("days_to_report", 0),
            exclusions_triggered=exclusions_triggered,
            deductible=policy["deductible"],
            coverage_limit=policy["coverage_limit"],
            waiting_period_active=waiting_period_active,
            applicable_state=policy["state"],
            findings=findings,
        )

        run_dir = self.runs_dir / claim_id

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._write_json(
            run_dir / "coverage_result.json",
            result.model_dump(mode="json"),
        )

        return result