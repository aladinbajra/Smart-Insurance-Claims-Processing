import json

from datetime import datetime
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

        self.policy_pack = load_policy(
            self.config_path
        )

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
    def _write_audit_log(
        path: Path,
        claim_id: str,
        entries: list[str],
    ) -> None:

        lines = [
            f"# Settlement Audit: {claim_id}",
            "",
        ]

        lines.extend(
            f"- {entry}"
            for entry in entries
        )

        path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _calculate_repair_variance(
        repair_estimate: float,
        market_value: float,
    ) -> float:

        if market_value <= 0:
            return 0.0

        return round(
            repair_estimate / market_value,
            4,
        )

    @staticmethod
    def _calculate_medical_variance(
        medical_total: float,
        gross_amount: float,
    ) -> float:

        if gross_amount <= 0:
            return 0.0

        return round(
            medical_total / gross_amount,
            4,
        )

    @staticmethod
    def _is_within_tolerance(
        value: float,
        tolerance: float,
    ) -> bool:

        return value <= tolerance

    def process(
        self,
        context_packet: dict,
    ) -> SettlementCalc:

        financials = context_packet["financials"]

        claim_id = context_packet["claim_id"]

        repair_estimate = financials["repair_estimate"]

        medical_total = financials["medical_total"]

        replacement_value = financials.get(
            "replacement_value",
            0.0,
        )

        deductible = financials["deductible"]

        market_value = financials["market_value"]

        findings: list[Finding] = []

        audit_entries: list[str] = []

        if market_value <= 0:
            findings.append(
                self._create_finding(
                    finding_id="D-006-MISSING_MARKET_VALUE",
                    severity=Severity.low,
                    recommendation=(
                        "Market value unavailable. "
                        "Variance calculations limited."
                    ),
                    evidence_links=[
                        "context_packet.financials.market_value",
                    ],
                    timestamp=datetime.utcnow().isoformat(),
                )
            )

            audit_entries.append(
                "Market value missing."
            )

        line_items = [
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

        if replacement_value > 0:
            line_items.append(
                LineItem(
                    description="Replacement Value",
                    amount=replacement_value,
                    category="replacement",
                )
            )

        gross_amount = (
            repair_estimate
            + medical_total
            + replacement_value
        )

        audit_entries.append(
            f"Gross amount calculated: {gross_amount}"
        )

        net_settlement = max(
            gross_amount - deductible,
            0.0,
        )

        audit_entries.append(
            f"Net settlement calculated: {net_settlement}"
        )

        total_loss_ratio = (
            gross_amount / market_value
            if market_value > 0
            else 0.0
        )

        total_loss_flag = (
            total_loss_ratio >= 0.75
        )

        if total_loss_flag:
            findings.append(
                self._create_finding(
                    finding_id="D-001-TOTAL_LOSS",
                    severity=Severity.high,
                    recommendation=(
                        "Escalate total loss settlement."
                    ),
                    evidence_links=[
                        "context_packet.financials.market_value",
                    ],
                    timestamp=datetime.utcnow().isoformat(),
                )
            )

            audit_entries.append(
                "Total loss detected."
            )

        if gross_amount <= deductible:
            findings.append(
                self._create_finding(
                    finding_id="D-002-NEGATIVE_SETTLEMENT",
                    severity=Severity.medium,
                    recommendation=(
                        "Review deductible application."
                    ),
                    evidence_links=[
                        "context_packet.financials.deductible",
                    ],
                    timestamp=datetime.utcnow().isoformat(),
                )
            )

            audit_entries.append(
                "Settlement reduced to zero."
            )

        repair_variance_pct = (
            self._calculate_repair_variance(
                repair_estimate,
                market_value,
            )
        )

        medical_variance_pct = (
            self._calculate_medical_variance(
                medical_total,
                gross_amount,
            )
        )

        repair_within_tolerance = (
            self._is_within_tolerance(
                repair_variance_pct,
                self.policy_pack.tolerances.repair_cost_variance_pct,
            )
        )

        medical_within_tolerance = (
            self._is_within_tolerance(
                medical_variance_pct,
                self.policy_pack.tolerances.medical_cost_variance_pct,
            )
        )

        if not repair_within_tolerance:
            findings.append(
                self._create_finding(
                    finding_id="D-003-REPAIR_VARIANCE_EXCEEDED",
                    severity=Severity.medium,
                    recommendation=(
                        "Review repair estimate variance."
                    ),
                    evidence_links=[
                        "context_packet.financials.repair_estimate",
                    ],
                    timestamp=datetime.utcnow().isoformat(),
                )
            )

            audit_entries.append(
                "Repair variance exceeded tolerance."
            )

        if not medical_within_tolerance:
            findings.append(
                self._create_finding(
                    finding_id="D-004-MEDICAL_VARIANCE_EXCEEDED",
                    severity=Severity.medium,
                    recommendation=(
                        "Review medical estimate variance."
                    ),
                    evidence_links=[
                        "context_packet.financials.medical_total",
                    ],
                    timestamp=datetime.utcnow().isoformat(),
                )
            )

            audit_entries.append(
                "Medical variance exceeded tolerance."
            )

        if replacement_value > market_value and market_value > 0:
            findings.append(
                self._create_finding(
                    finding_id="D-005-REPLACEMENT_EXCEEDS_MARKET",
                    severity=Severity.medium,
                    recommendation=(
                        "Review replacement value against market value."
                    ),
                    evidence_links=[
                        "context_packet.financials.replacement_value",
                        "context_packet.financials.market_value",
                    ],
                    timestamp=datetime.utcnow().isoformat(),
                )
            )

            audit_entries.append(
                "Replacement value exceeds market value."
            )

        audit_entries.append(
            f"Settlement findings generated: {len(findings)}"
        )

        audit_entries.append(
            "Settlement processing completed."
        )

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
            total_loss_ratio=round(
                total_loss_ratio,
                4,
            ),
            total_loss_flag=total_loss_flag,
            repair_variance_pct=repair_variance_pct,
            medical_variance_pct=medical_variance_pct,
            repair_within_tolerance=repair_within_tolerance,
            medical_within_tolerance=medical_within_tolerance,
            findings=findings,
            finalized=False,
        )

        run_dir = self.runs_dir / claim_id

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._write_json(
            run_dir / "settlement_calc.json",
            result.model_dump(mode="json"),
        )

        self._write_audit_log(
            run_dir / "settlement_audit.md",
            claim_id,
            audit_entries,
        )

        return result