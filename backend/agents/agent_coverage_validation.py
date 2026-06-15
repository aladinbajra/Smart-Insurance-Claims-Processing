from datetime import datetime

from backend.models.schemas import (
    CoverageResult,
    Finding,
    Severity,
)


class AgentCoverageValidation:
    """
    Agent C: Coverage & Policy Validation.

    Responsibilities:
    - Verify policy status.
    - Detect expired policies.
    - Detect coverage denials.
    - Detect late notice violations.
    - Produce coverage findings.
    """

    def process(self, context_packet: dict) -> CoverageResult:
        findings = []

        policy = context_packet["policy"]
        incident = context_packet["incident"]
        flags = context_packet["flags"]

        coverage_active = True
        denial_reason = None
        exclusions_triggered = []
        waiting_period_active = False

        # ---------------------------------------------------
        # Rule 1: Policy Expired
        # ---------------------------------------------------
        if policy["policy_expired"]:
            coverage_active = False
            denial_reason = "Policy expired"

            findings.append(
                Finding(
                    finding_id="C-001-POLICY_EXPIRED",
                    agent="agent_coverage_validation",
                    severity=Severity.high,
                    confidence=1.0,
                    evidence_links=[
                        "context_packet.policy.policy_expired"
                    ],
                    recommendation=(
                        "Deny claim and escalate for coverage review."
                    ),
                    open_questions=[],
                    requires_human_review=True,
                    timestamp=datetime.utcnow().isoformat(),
                )
            )

        # ---------------------------------------------------
        # Rule 2: Coverage Denied
        # ---------------------------------------------------
        elif flags["coverage_denied"]:
            coverage_active = False
            denial_reason = "Coverage denied"

            findings.append(
                Finding(
                    finding_id="C-002-COVERAGE_DENIED",
                    agent="agent_coverage_validation",
                    severity=Severity.high,
                    confidence=1.0,
                    evidence_links=[
                        "context_packet.flags.coverage_denied"
                    ],
                    recommendation=(
                        "Coverage does not apply to this claim."
                    ),
                    open_questions=[],
                    requires_human_review=True,
                    timestamp=datetime.utcnow().isoformat(),
                )
            )

        # ---------------------------------------------------
        # Rule 3: Late Notice
        # ---------------------------------------------------
        late_notice_violation = (
            incident["days_to_report"]
            > policy["late_notice_days"]
        )

        if late_notice_violation:
            findings.append(
                Finding(
                    finding_id="C-003-LATE_NOTICE",
                    agent="agent_coverage_validation",
                    severity=Severity.medium,
                    confidence=1.0,
                    evidence_links=[
                        "context_packet.incident.days_to_report"
                    ],
                    recommendation=(
                        "Review late reporting before settlement."
                    ),
                    open_questions=[],
                    requires_human_review=True,
                    timestamp=datetime.utcnow().isoformat(),
                )
            )

        return CoverageResult(
            claim_id=context_packet["claim_id"],
            coverage_active=coverage_active,
            policy_expired=policy["policy_expired"],
            denial_reason=denial_reason,
            late_notice_violation=late_notice_violation,
            late_notice_days_allowed=policy["late_notice_days"],
            days_to_report=incident["days_to_report"],
            exclusions_triggered=exclusions_triggered,
            deductible=policy["deductible"],
            coverage_limit=policy["coverage_limit"],
            waiting_period_active=waiting_period_active,
            applicable_state=policy["state"],
            findings=findings,
        )