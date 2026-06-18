from __future__ import annotations

import csv
import json
from pathlib import Path

from backend.agents.agent_coverage_validation import AgentCoverageValidation
from backend.pipeline.runner import PipelineRunner


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config() -> Path:
    return _root() / "config" / "policy.yaml"


def _dataset(scenario: str) -> Path:
    return _root() / "datasets" / "SICPS_Dataset (3)" / "SICPS_Dataset" / scenario


def _runner(tmp_path: Path) -> PipelineRunner:
    return PipelineRunner(config_path=_config(), runs_dir=tmp_path / "runs")


# ---------------------------------------------------------------------------
# REQ-004 — CoreLogic-ready CSV settlement payload
# ---------------------------------------------------------------------------

def test_csv_settlement_payload(tmp_path: Path) -> None:
    result = _runner(tmp_path).run(_dataset("scenario_01_clean_auto"))
    csv_path = result.run_dir / "settlement_payload.csv"
    assert csv_path.exists()

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert row["claim_id"] == "CLM-2026-001"
    assert row["routing_decision"] == "auto_settle"
    assert float(row["net_settlement"]) == 550.0
    assert row["invoice_match_status"] == "match"


# ---------------------------------------------------------------------------
# REQ-019 — total loss pays the market / replacement value, not repair cost
# ---------------------------------------------------------------------------

def test_total_loss_pays_replacement_value(tmp_path: Path) -> None:
    result = _runner(tmp_path).run(_dataset("scenario_02_total_loss"))
    settlement = json.loads((result.run_dir / "settlement_calc.json").read_text())

    # market 32000 + medical 4200 = 36200 gross (NOT repair 34600 based).
    assert settlement["total_loss_flag"] is True
    assert settlement["replacement_value"] == 32000.0
    assert settlement["gross_amount"] == 36200.0
    assert settlement["net_settlement"] == 34200.0  # minus 2000 deductible

    categories = {li["category"] for li in settlement["line_items"]}
    assert "replacement" in categories
    assert "repair" not in categories  # repair cost is not paid on a total loss


# ---------------------------------------------------------------------------
# REQ-037 — regional state overrides change the late-notice window
# ---------------------------------------------------------------------------

def _claim(incident_date: str, days: int, medical: float = 0.0) -> dict:
    return {
        "claim_id": "TEST-IN-001",
        "incident_date": incident_date,
        "days_to_report": days,
        "medical_total": medical,
        "extracted_fields": [],
    }


def _policy(state: str) -> dict:
    return {
        "effective_date": "2024-01-01",
        "expiration_date": "2027-01-01",
        "exclusions": [],
        "deductible": 500.0,
        "coverage_limit": 50000.0,
        "state": state,
        "late_notice_days": 30,
        "pre_existing_lookback_days": 0,
    }


def test_state_override_extends_late_notice_window(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    claim = _claim("2026-05-01", days=40)  # 40 days: > 30, but <= IN override 45
    policy = _policy("IN")

    # Without config -> no overrides -> 40 > 30 -> violation.
    plain = AgentCoverageValidation(runs).process(claim, policy)
    assert plain.late_notice_violation is True
    assert plain.late_notice_days_allowed == 30

    # With config -> IN override 45 -> 40 <= 45 -> no violation.
    with_cfg = AgentCoverageValidation(runs, _config()).process(claim, policy)
    assert with_cfg.late_notice_violation is False
    assert with_cfg.late_notice_days_allowed == 45


# ---------------------------------------------------------------------------
# REQ-017 — coverage sub-limits flag an over-cap medical payout
# ---------------------------------------------------------------------------

def test_medical_sub_limit_exceeded(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    claim = _claim("2026-05-01", days=1, medical=60000.0)  # > 50000 sub-limit
    result = AgentCoverageValidation(runs, _config()).process(claim, _policy("IL"))

    codes = [f.finding_id for f in result.findings]
    assert "C-005-SUBLIMIT_EXCEEDED" in codes


def test_medical_within_sub_limit_no_flag(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    claim = _claim("2026-05-01", days=1, medical=4200.0)  # < 50000 sub-limit
    result = AgentCoverageValidation(runs, _config()).process(claim, _policy("IL"))

    codes = [f.finding_id for f in result.findings]
    assert "C-005-SUBLIMIT_EXCEEDED" not in codes
