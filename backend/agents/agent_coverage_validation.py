import json

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    ) -> None:

        self.runs_dir = Path(runs_dir)

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

    @staticmethod
    def _now() -> str:

        return datetime.now(timezone.utc).isoformat()

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
            timestamp=self._now(),
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
        incident: dict = claim_summary["incident"]
        claim_details: dict = claim_summary.get("claim_details", {})

        coverage_active: bool = True
        denial_reason: str | None = None
        exclusions_triggered: list[str] = []
        waiting_period_active: bool = False

        # ----------------------------------------------------------
        # Derived dates
        # ----------------------------------------------------------
        incident_date = self._parse_date(incident["date"])
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
                        "claim_summary.incident.date",
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
                        "claim_summary.incident.date",
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

        claim_text_fields: list[str] = [
            str(v).lower()
            for v in claim_details.values()
            if isinstance(v, str)
        ]

        incident_description: str = (
            incident.get("description", "").lower()
        )

        for exclusion in policy_exclusions:
            exclusion_lower = exclusion.lower()

            matched = any(
                exclusion_lower in field
                for field in claim_text_fields
            ) or exclusion_lower in incident_description

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
                        "claim_summary.claim_details",
                        "claim_summary.incident.description",
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
                            "claim_summary.incident.date",
                            "policy.effective_date",
                        ],
                    )
                )

        # ----------------------------------------------------------
        # Rule 5: Late Notice
        # ----------------------------------------------------------
        days_to_report: int = incident.get("days_to_report", 0)
        late_notice_days: int = policy.get("late_notice_days", 0)

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
                        "claim_summary.incident.days_to_report",
                        "policy.late_notice_days",
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
    ) -> CoverageResult:

        incident: dict = claim_summary["incident"]

        result = CoverageResult(
            claim_id=claim_id,
            coverage_active=coverage_active,
            policy_expired=policy_expired,
            denial_reason=denial_reason,
            late_notice_violation=late_notice_violation,
            late_notice_days_allowed=policy.get("late_notice_days", 0),
            days_to_report=days_to_report or incident.get("days_to_report", 0),
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