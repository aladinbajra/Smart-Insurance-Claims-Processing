from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from backend.models.schemas import (
    Finding,
    FraudIndicator,
    FraudScore,
    RiskLevel,
    Severity,
)


class AgentFraudDetection:
    """
    Agent E: Fraud Detection Engine.

    This agent reads the structured claim context created by Agent A
    and optional outputs from Agents B, C, and D when they are available.

    It generates fraud_score.json with:
    - fraud score
    - risk level
    - SIU referral decision
    - adjuster review decision
    - fraud indicators
    - evidence links
    - findings for audit traceability
    """

    def __init__(
        self,
        config_path: str | Path,
    ) -> None:
        # Store the shared policy configuration path.
        self.config_path = Path(config_path)

        # Load fraud thresholds and known-risk entities from policy.yaml.
        self.policy = self._load_yaml(self.config_path)

        self.fraud_policy = self.policy.get("fraud", {})

    def process(
        self,
        run_dir: str | Path,
    ) -> dict[str, Any]:
        """
        Process one claim run folder.

        Expected input:
        - context_packet.json from Agent A

        Optional inputs:
        - claim_summary.json from Agent B
        - coverage_result.json from Agent C
        - settlement_calc.json from Agent D

        Output:
        - fraud_score.json
        """

        run_path = Path(run_dir)

        context_path = run_path / "context_packet.json"

        if not context_path.is_file():
            raise FileNotFoundError(
                f"context_packet.json not found: {context_path}"
            )

        context = self._load_json(context_path)

        # Optional downstream files. These may not exist yet while
        # other team members are still working on Agents B, C, and D.
        claim_summary = self._load_optional_json(
            run_path / "claim_summary.json"
        )

        coverage_result = self._load_optional_json(
            run_path / "coverage_result.json"
        )

        settlement_calc = self._load_optional_json(
            run_path / "settlement_calc.json"
        )

        indicators: list[FraudIndicator] = []
        findings: list[Finding] = []

        claim_id = context["claim_id"]
        created_at = context.get(
            "created_at",
            "1970-01-01T00:00:00Z",
        )

        claimant = context.get("claimant", {}) or {}
        provider = context.get("provider", {}) or {}
        financials = context.get("financials", {}) or {}
        flags = context.get("flags", {}) or {}

        # Detect fraud indicators from Agent A context.
        self._check_claimant_history(
            claimant=claimant,
            indicators=indicators,
        )

        self._check_provider_risk(
            provider=provider,
            indicators=indicators,
        )

        self._check_claim_flags(
            flags=flags,
            indicators=indicators,
        )

        self._check_financial_risk(
            financials=financials,
            settlement_calc=settlement_calc,
            indicators=indicators,
        )

        self._check_optional_agent_outputs(
            claim_summary=claim_summary,
            coverage_result=coverage_result,
            indicators=indicators,
        )

        fraud_score = self._calculate_fraud_score(indicators)

        risk_level = self._determine_risk_level(fraud_score)

        provider_blacklisted = self._is_provider_blacklisted(provider)

        provider_ring_id = self._matched_provider_ring_id(provider)

        provider_ring_match = provider_ring_id is not None

        prior_claims_trigger = self._has_prior_claims_trigger(claimant)

        billing_variance_pct = self._billing_variance_pct(
            settlement_calc=settlement_calc
        )

        siu_referral = (
            fraud_score
            >= float(
                self.fraud_policy.get("high_risk_threshold", 0.75)
            )
            or provider_blacklisted
            or provider_ring_match
            or bool(claimant.get("fraud_history", False))
            or bool(flags.get("duplicate_claim", False))
        )

        adjuster_review = (
            siu_referral
            or fraud_score
            >= float(
                self.fraud_policy.get("medium_risk_threshold", 0.40)
            )
            or bool(flags.get("missing_documents", False))
        )

        evidence_links = [
            indicator.evidence
            for indicator in indicators
        ]

        findings = self._build_findings(
            claim_id=claim_id,
            indicators=indicators,
            fraud_score=fraud_score,
            risk_level=risk_level,
            siu_referral=siu_referral,
            created_at=created_at,
        )

        fraud_result = FraudScore(
            claim_id=claim_id,
            fraud_score=fraud_score,
            risk_level=risk_level,
            siu_referral=siu_referral,
            adjuster_review=adjuster_review,
            provider_blacklisted=provider_blacklisted,
            provider_ring_match=provider_ring_match,
            provider_ring_id=provider_ring_id,
            prior_claims_trigger=prior_claims_trigger,
            billing_variance_pct=billing_variance_pct,
            indicators=indicators,
            evidence_links=evidence_links,
            findings=findings,
        )

        self._write_json(
            run_path / "fraud_score.json",
            fraud_result.model_dump(mode="json"),
        )

        return fraud_result.model_dump(mode="json")

    def _check_claimant_history(
        self,
        claimant: dict[str, Any],
        indicators: list[FraudIndicator],
    ) -> None:
        """Check claimant-history fraud signals."""

        if claimant.get("fraud_history") is True:
            indicators.append(
                FraudIndicator(
                    indicator="prior_fraud_history",
                    weight=0.35,
                    evidence="context_packet.json:claimant.fraud_history",
                )
            )

        if claimant.get("new_claimant") is True:
            indicators.append(
                FraudIndicator(
                    indicator="new_claimant",
                    weight=0.10,
                    evidence="context_packet.json:claimant.new_claimant",
                )
            )

        if self._has_prior_claims_trigger(claimant):
            indicators.append(
                FraudIndicator(
                    indicator="multiple_prior_claims",
                    weight=0.25,
                    evidence=(
                        "context_packet.json:"
                        "claimant.prior_claims_18mo"
                    ),
                )
            )

    def _check_provider_risk(
        self,
        provider: dict[str, Any],
        indicators: list[FraudIndicator],
    ) -> None:
        """Check provider blacklists and fraud-ring matches."""

        if not provider:
            return

        if self._is_provider_blacklisted(provider):
            indicators.append(
                FraudIndicator(
                    indicator="blacklisted_provider",
                    weight=0.40,
                    evidence="context_packet.json:provider",
                )
            )

        ring_id = self._matched_provider_ring_id(provider)

        if ring_id:
            indicators.append(
                FraudIndicator(
                    indicator="known_fraud_ring_match",
                    weight=0.35,
                    evidence=(
                        f"context_packet.json:provider."
                        f"fraud_ring_id={ring_id}"
                    ),
                )
            )

    def _check_claim_flags(
        self,
        flags: dict[str, Any],
        indicators: list[FraudIndicator],
    ) -> None:
        """Check fraud-related claim flags."""

        if flags.get("duplicate_claim") is True:
            indicators.append(
                FraudIndicator(
                    indicator="duplicate_claim",
                    weight=0.30,
                    evidence="context_packet.json:flags.duplicate_claim",
                )
            )

        if flags.get("missing_documents") is True:
            indicators.append(
                FraudIndicator(
                    indicator="missing_documents",
                    weight=0.20,
                    evidence="context_packet.json:flags.missing_documents",
                )
            )

        if flags.get("low_ocr_confidence") is True:
            indicators.append(
                FraudIndicator(
                    indicator="low_ocr_confidence",
                    weight=0.10,
                    evidence="context_packet.json:flags.low_ocr_confidence",
                )
            )

        if flags.get("siu_referral") is True:
            indicators.append(
                FraudIndicator(
                    indicator="manifest_siu_referral_flag",
                    weight=0.30,
                    evidence="context_packet.json:flags.siu_referral",
                )
            )

    def _check_financial_risk(
        self,
        financials: dict[str, Any],
        settlement_calc: dict[str, Any],
        indicators: list[FraudIndicator],
    ) -> None:
        """Check inflated repair and total-loss fraud signals."""

        if financials.get("total_loss_flag") is True:
            indicators.append(
                FraudIndicator(
                    indicator="total_loss_claim",
                    weight=0.15,
                    evidence="context_packet.json:financials.total_loss_flag",
                )
            )

        repair_variance_pct = settlement_calc.get(
            "repair_variance_pct"
        )

        if repair_variance_pct is not None:
            inflated_threshold = float(
                self.fraud_policy.get(
                    "inflated_repair_multiplier",
                    1.50,
                )
            )

            if float(repair_variance_pct) > inflated_threshold:
                indicators.append(
                    FraudIndicator(
                        indicator="inflated_repair_cost",
                        weight=0.25,
                        evidence=(
                            "settlement_calc.json:"
                            "repair_variance_pct"
                        ),
                    )
                )

    def _check_optional_agent_outputs(
        self,
        claim_summary: dict[str, Any],
        coverage_result: dict[str, Any],
        indicators: list[FraudIndicator],
    ) -> None:
        """
        Use Agent B and Agent C outputs if they already exist.

        Agent E still works when these files are missing.
        """

        if claim_summary.get("requires_manual_review") is True:
            indicators.append(
                FraudIndicator(
                    indicator="document_extraction_manual_review",
                    weight=0.10,
                    evidence="claim_summary.json:requires_manual_review",
                )
            )

        if coverage_result.get("late_notice_violation") is True:
            indicators.append(
                FraudIndicator(
                    indicator="late_notice_violation",
                    weight=0.15,
                    evidence="coverage_result.json:late_notice_violation",
                )
            )

        if coverage_result.get("policy_expired") is True:
            indicators.append(
                FraudIndicator(
                    indicator="expired_policy",
                    weight=0.25,
                    evidence="coverage_result.json:policy_expired",
                )
            )

    def _has_prior_claims_trigger(
        self,
        claimant: dict[str, Any],
    ) -> bool:
        """Check whether prior-claim count reaches the SIU trigger."""

        prior_claims = int(
            claimant.get("prior_claims_18mo", 0)
        )

        threshold = int(
            self.fraud_policy.get(
                "prior_claims_siu_trigger",
                3,
            )
        )

        return prior_claims >= threshold

    def _is_provider_blacklisted(
        self,
        provider: dict[str, Any],
    ) -> bool:
        """Check provider against the policy blacklist."""

        if provider.get("is_blacklisted") is True:
            return True

        provider_id = provider.get("provider_id")
        tax_id = provider.get("tax_id")

        for blacklisted_provider in self.fraud_policy.get(
            "blacklisted_providers",
            []
        ):
            if provider_id and provider_id == blacklisted_provider.get("id"):
                return True

            if tax_id and tax_id == blacklisted_provider.get("tax_id"):
                return True

        return False

    def _matched_provider_ring_id(
        self,
        provider: dict[str, Any],
    ) -> str | None:
        """Check whether the provider belongs to a known fraud ring."""

        if provider.get("fraud_ring_id"):
            return str(provider["fraud_ring_id"])

        provider_id = provider.get("provider_id")
        tax_id = provider.get("tax_id")

        for ring in self.fraud_policy.get(
            "known_fraud_rings",
            []
        ):
            if provider_id in ring.get("provider_ids", []):
                return str(ring.get("ring_id"))

            if tax_id in ring.get("tax_ids", []):
                return str(ring.get("ring_id"))

        return None

    @staticmethod
    def _billing_variance_pct(
        settlement_calc: dict[str, Any],
    ) -> float:
        """Read billing variance from Agent D output if available."""

        value = settlement_calc.get(
            "repair_variance_pct",
            0.0,
        )

        return float(value or 0.0)

    @staticmethod
    def _calculate_fraud_score(
        indicators: list[FraudIndicator],
    ) -> float:
        """
        Calculate fraud probability from weighted indicators.

        The score is capped at 1.0.
        """

        score = sum(
            indicator.weight
            for indicator in indicators
        )

        return min(round(score, 4), 1.0)

    def _determine_risk_level(
        self,
        fraud_score: float,
    ) -> RiskLevel:
        """Map the numeric fraud score to a risk level."""

        high_threshold = float(
            self.fraud_policy.get("high_risk_threshold", 0.75)
        )

        medium_threshold = float(
            self.fraud_policy.get("medium_risk_threshold", 0.40)
        )

        if fraud_score >= 0.90:
            return RiskLevel.critical

        if fraud_score >= high_threshold:
            return RiskLevel.high

        if fraud_score >= medium_threshold:
            return RiskLevel.medium

        return RiskLevel.low

    @staticmethod
    def _build_findings(
        claim_id: str,
        indicators: list[FraudIndicator],
        fraud_score: float,
        risk_level: RiskLevel,
        siu_referral: bool,
        created_at: str,
    ) -> list[Finding]:
        """Create audit-ready findings from detected fraud indicators."""

        findings: list[Finding] = []

        for index, indicator in enumerate(
            indicators,
            start=1,
        ):
            severity = (
                Severity.high
                if indicator.weight >= 0.30
                else Severity.medium
            )

            findings.append(
                Finding(
                    finding_id=(
                        f"E-{index:03d}-{indicator.indicator}"
                    ),
                    agent="agent_fraud_detection",
                    severity=severity,
                    confidence=min(
                        round(indicator.weight + fraud_score, 4),
                        1.0,
                    ),
                    evidence_links=[indicator.evidence],
                    recommendation=(
                        "Refer claim to SIU for fraud review."
                        if siu_referral
                        else "Continue with adjuster review if needed."
                    ),
                    open_questions=[],
                    requires_human_review=siu_referral,
                    timestamp=created_at,
                )
            )

        if not findings and risk_level == RiskLevel.low:
            findings.append(
                Finding(
                    finding_id=f"E-000-{claim_id}-LOW-RISK",
                    agent="agent_fraud_detection",
                    severity=Severity.info,
                    confidence=1.0,
                    evidence_links=["context_packet.json"],
                    recommendation=(
                        "No fraud indicators detected. "
                        "Continue normal claim processing."
                    ),
                    open_questions=[],
                    requires_human_review=False,
                    timestamp=created_at,
                )
            )

        return findings

    @staticmethod
    def _load_yaml(
        path: Path,
    ) -> dict[str, Any]:
        """Load YAML file content."""

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        return data or {}

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict[str, Any]:
        """Load JSON file content."""

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _load_optional_json(
        path: Path,
    ) -> dict[str, Any]:
        """Load JSON only if the file exists."""

        if not path.is_file():
            return {}

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _write_json(
        path: Path,
        data: dict[str, Any],
    ) -> None:
        """Write deterministic JSON output."""

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=2,
                sort_keys=True,
            )

            file.write("\n")