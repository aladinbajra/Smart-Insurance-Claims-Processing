import json

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.models.schemas import (
    SettlementCalc,
    LineItem,
    Finding,
    Severity,
)
from backend.tools.pdf_extractor import load_policy


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

    @staticmethod
    def _now() -> str:
        # Fix D-1: replaces datetime.utcnow().isoformat()
        return datetime.now(timezone.utc).isoformat()

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
                    timestamp=self._now(),
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

        # ClaimSummary schema does not carry injury_claim or replacement_value
        injury_claim: float = 0.0
        replacement_value: float = 0.0

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
                    timestamp=self._now(),
                )
            )
            audit_entries.append("Market value missing.")

        # --------------------------------------------------------------
        # Line items
        # --------------------------------------------------------------
        line_items: list[LineItem] = [
            LineItem(
                description="Repair Estimate",
                amount=repair_estimate,
                category="repair",
            ),
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

        if replacement_value > 0:
            line_items.append(
                LineItem(
                    description="Replacement Value",
                    amount=replacement_value,
                    category="replacement",
                )
            )

        # --------------------------------------------------------------
        # Gross amount
        # --------------------------------------------------------------
        gross_amount: float = (
            repair_estimate + medical_total + injury_claim + replacement_value
        )

        audit_entries.append(f"Gross amount calculated (Agent D): {gross_amount}")

        # --------------------------------------------------------------
        # Net settlement
        # --------------------------------------------------------------
        net_settlement: float = max(gross_amount - deductible, 0.0)

        audit_entries.append(f"Net settlement calculated: {net_settlement}")

        # --------------------------------------------------------------
        # Total loss detection
        # Fix D-2: ratio uses repair_estimate / market_value only
        # Fix D-3: threshold from policy_pack, not hardcoded 0.75
        # --------------------------------------------------------------
        total_loss_ratio: float = (
            repair_estimate / market_value if market_value > 0 else 0.0
        )

        total_loss_threshold: float = (
            self.policy_pack.approval.total_loss_ratio_threshold
        )

        total_loss_flag: bool = total_loss_ratio >= total_loss_threshold

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
                    timestamp=self._now(),
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
                    timestamp=self._now(),
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
                    timestamp=self._now(),
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
                    timestamp=self._now(),
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
                    timestamp=self._now(),
                )
            )
            audit_entries.append("Replacement value exceeds market value.")

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
