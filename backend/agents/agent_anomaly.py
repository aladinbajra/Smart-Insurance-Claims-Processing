from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from backend.models.schemas import Finding, Severity


class AgentAnomaly:
    """
    Anomaly Detection Agent (stretch REQ-052): outlier + surge flags.

    Deterministic, threshold-based anomaly screening that runs after damage and
    fraud assessment. It flags statistical/business outliers (high gross,
    extreme repair-to-market ratio, large billing variance) and surge
    conditions (catastrophe events). Advisory — it adds findings but does not
    change routing. Rule-based, so it stays reproducible (REQ-044).

    Output: runs/<claim_id>/anomaly_report.json
    """

    def __init__(self, config_path: str | Path, runs_dir: str | Path) -> None:
        self.runs_dir = Path(runs_dir)
        cfg = self._load_yaml(Path(config_path))
        self.approval = cfg.get("approval", {}) or {}
        self.tolerances = cfg.get("tolerances", {}) or {}

    def process(self, run_dir: str | Path) -> dict[str, Any]:
        run_path = Path(run_dir)
        context = self._load(run_path / "context_packet.json")
        settlement = self._load(run_path / "settlement_calc.json")

        claim_id = str(context.get("claim_id", run_path.name))
        created_at = str(context.get("created_at", "1970-01-01T00:00:00Z"))
        incident = context.get("incident", {}) or {}

        gross = self._f(settlement.get("gross_amount"))
        total_loss_ratio = self._f(settlement.get("total_loss_ratio"))
        total_loss_flag = bool(settlement.get("total_loss_flag", False))
        billing_variance = self._f(settlement.get("billing_variance_pct"))

        senior_threshold = float(self.approval.get("senior_review_threshold", 50000.0))
        repair_tol = float(self.tolerances.get("repair_cost_variance_pct", 0.25))

        anomalies: list[dict[str, Any]] = []
        findings: list[Finding] = []

        def flag(code: str, detail: str, severity: Severity, evidence: str) -> None:
            anomalies.append({"code": code, "detail": detail})
            findings.append(
                Finding(
                    finding_id=f"AN-{len(findings) + 1:03d}-{code}",
                    agent="agent_anomaly",
                    severity=severity,
                    confidence=0.8,
                    evidence_links=[evidence],
                    recommendation=f"Anomaly: {detail}",
                    open_questions=[],
                    requires_human_review=False,
                    timestamp=created_at,
                )
            )

        if gross > senior_threshold:
            flag("HIGH_VALUE_OUTLIER",
                 f"gross {gross:,.2f} exceeds senior threshold {senior_threshold:,.2f}",
                 Severity.medium, "settlement_calc.json:gross_amount")

        # Only an anomaly if repair exceeds value WITHOUT being recognized as a
        # total loss — that is an inconsistency. A flagged total loss is expected,
        # not anomalous (it is already handled by Agent D).
        if total_loss_ratio >= 1.0 and not total_loss_flag:
            flag("REPAIR_EXCEEDS_VALUE",
                 f"repair-to-market ratio {total_loss_ratio} >= 1.0 but not flagged total loss",
                 Severity.medium, "settlement_calc.json:total_loss_ratio")

        if billing_variance > repair_tol:
            flag("BILLING_VARIANCE_OUTLIER",
                 f"billing variance {billing_variance} exceeds tolerance {repair_tol}",
                 Severity.medium, "settlement_calc.json:billing_variance_pct")

        if incident.get("cat_event") is True:
            flag("CAT_SURGE",
                 "claim falls under an active catastrophe event (surge conditions)",
                 Severity.low, "context_packet.json:incident.cat_event")

        # Deterministic anomaly score: capped count of distinct anomalies.
        anomaly_score = round(min(len(anomalies) * 0.25, 1.0), 4)

        if not anomalies:
            findings.append(
                Finding(
                    finding_id="AN-000-NO_ANOMALY",
                    agent="agent_anomaly",
                    severity=Severity.info,
                    confidence=1.0,
                    evidence_links=["settlement_calc.json"],
                    recommendation="No statistical or surge anomalies detected.",
                    open_questions=[],
                    requires_human_review=False,
                    timestamp=created_at,
                )
            )

        result = {
            "claim_id": claim_id,
            "anomaly_score": anomaly_score,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
            "surge_active": bool(incident.get("cat_event", False)),
            "findings": [f.model_dump(mode="json") for f in findings],
        }
        self._write_json(run_path / "anomaly_report.json", result)
        return result

    @staticmethod
    def _f(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
            file.write("\n")
