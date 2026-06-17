from __future__ import annotations

import json
from pathlib import Path

import yaml

from backend.agents.agent_coverage_validation import AgentCoverageValidation
from backend.agents.agent_damage_assessment import AgentDamageAssessment
from backend.agents.agent_document_extraction import AgentDocumentExtraction
from backend.agents.agent_fnol_intake import AgentFNOLIntake


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dataset(scenario: str) -> Path:
    return (
        _root()
        / "datasets"
        / "SICPS_Dataset (3)"
        / "SICPS_Dataset"
        / scenario
    )


def _config() -> Path:
    return _root() / "config" / "policy.yaml"


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _run_through_c(runs_dir: Path, scenario: str, claim_id: str) -> None:
    """Run A -> B -> C so claim_summary.json and coverage_result.json exist."""
    AgentFNOLIntake(config_path=_config(), runs_dir=runs_dir).process(
        _dataset(scenario)
    )
    AgentDocumentExtraction(config_path=_config(), runs_dir=runs_dir).process(
        _dataset(scenario)
    )
    run_dir = runs_dir / claim_id
    claim_summary = _load(run_dir / "claim_summary.json")
    policy = _load(run_dir / "context_packet.json")["policy"]
    AgentCoverageValidation(runs_dir=runs_dir).process(claim_summary, policy)


def _run_damage(runs_dir: Path, claim_id: str, config: Path | None = None) -> dict:
    run_dir = runs_dir / claim_id
    claim_summary = _load(run_dir / "claim_summary.json")
    coverage_result = _load(run_dir / "coverage_result.json")
    agent_d = AgentDamageAssessment(
        config_path=config or _config(),
        runs_dir=runs_dir,
    )
    return agent_d.process(claim_summary, coverage_result).model_dump(mode="json")


def _finding_codes(result: dict) -> list[str]:
    return [f["finding_id"] for f in result["findings"]]


# ---------------------------------------------------------------------------
# Scenario 01 — clean settlement, deductible sourced from coverage_result
# ---------------------------------------------------------------------------

def test_scenario_01_clean_settlement(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _run_through_c(runs_dir, "scenario_01_clean_auto", "CLM-2026-001")
    result = _run_damage(runs_dir, "CLM-2026-001")

    assert result["claim_id"] == "CLM-2026-001"
    assert result["repair_estimate"] == 1550.0
    assert result["medical_total"] == 0.0
    assert result["deductible"] == 1000.0
    assert result["gross_amount"] == 1550.0
    assert result["net_settlement"] == 550.0
    assert result["total_loss_flag"] is False

    # settlement_calc.json must be written to the run folder.
    assert (runs_dir / "CLM-2026-001" / "settlement_calc.json").exists()
    assert (runs_dir / "CLM-2026-001" / "settlement_audit.md").exists()


# ---------------------------------------------------------------------------
# Scenario 02 — total loss: ratio crosses the configured threshold
# ---------------------------------------------------------------------------

def test_scenario_02_total_loss(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _run_through_c(runs_dir, "scenario_02_total_loss", "CLM-2026-002")
    result = _run_damage(runs_dir, "CLM-2026-002")

    assert result["total_loss_flag"] is True
    # BUG-D1: ratio is repair / market only (medical excluded).
    assert result["total_loss_ratio"] == round(34600.0 / 32000.0, 4)
    assert "D-001-TOTAL_LOSS" in _finding_codes(result)


# ---------------------------------------------------------------------------
# BUG-D1 regression — scenario 07: repair/market < 0.75 but
# (repair + medical)/market >= 0.75. Medical must NOT inflate the ratio.
# ---------------------------------------------------------------------------

def test_scenario_07_medical_excluded_from_total_loss(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _run_through_c(runs_dir, "scenario_07_fraud_ring", "CLM-2026-007")
    result = _run_damage(runs_dir, "CLM-2026-007")

    # repair/market = 26300/38000 = 0.6921  -> NOT total loss
    # buggy gross/market would be (26300+9800)/38000 = 0.95 -> false positive
    assert result["total_loss_ratio"] == round(26300.0 / 38000.0, 4)
    assert result["total_loss_flag"] is False
    assert "D-001-TOTAL_LOSS" not in _finding_codes(result)


# ---------------------------------------------------------------------------
# BUG-D2 regression — total-loss threshold is read from policy.yaml, so a
# config change alone flips the decision (REQ-058 configurability).
# ---------------------------------------------------------------------------

def test_total_loss_threshold_is_configurable(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _run_through_c(runs_dir, "scenario_01_clean_auto", "CLM-2026-001")

    # scenario_01 ratio = 1550/24000 = 0.0646 — never a total loss at 0.75.
    baseline = _run_damage(runs_dir, "CLM-2026-001")
    assert baseline["total_loss_flag"] is False

    # Rewrite the threshold below the claim's ratio and re-run with that config.
    with _config().open("r", encoding="utf-8") as fh:
        policy_cfg = yaml.safe_load(fh)
    policy_cfg["approval"]["total_loss_ratio_threshold"] = 0.05
    custom_config = tmp_path / "policy_low_threshold.yaml"
    with custom_config.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(policy_cfg, fh)

    adjusted = _run_damage(runs_dir, "CLM-2026-001", config=custom_config)
    assert adjusted["total_loss_flag"] is True


# ---------------------------------------------------------------------------
# BUG-X1 regression — Agent D consumes claim_summary.json (financials) and
# coverage_result.json (deductible), not the manifest packet.
# ---------------------------------------------------------------------------

def test_consumes_claim_summary_and_coverage_result(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _run_through_c(runs_dir, "scenario_01_clean_auto", "CLM-2026-001")
    run_dir = runs_dir / "CLM-2026-001"

    # Tamper with Agent B's repair estimate and Agent C's deductible on disk.
    summary = _load(run_dir / "claim_summary.json")
    summary["repair_estimate"] = 8000.0
    with (run_dir / "claim_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh)

    coverage = _load(run_dir / "coverage_result.json")
    coverage["deductible"] = 2500.0
    with (run_dir / "coverage_result.json").open("w", encoding="utf-8") as fh:
        json.dump(coverage, fh)

    result = _run_damage(runs_dir, "CLM-2026-001")

    # Manifest repair_estimate is 1550 / deductible 1000; the tampered files win.
    assert result["repair_estimate"] == 8000.0
    assert result["deductible"] == 2500.0
    assert result["gross_amount"] == 8000.0
    assert result["net_settlement"] == 5500.0


# ---------------------------------------------------------------------------
# Scenario 08 — low OCR confidence halts settlement with a manual-review finding
# ---------------------------------------------------------------------------

def test_scenario_08_low_ocr_halts(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _run_through_c(runs_dir, "scenario_08_low_ocr", "CLM-2026-008")
    result = _run_damage(runs_dir, "CLM-2026-008")

    assert result["net_settlement"] == 0.0
    assert result["gross_amount"] == 0.0
    assert "D-007-UNRESOLVABLE_FILE" in _finding_codes(result)


# ---------------------------------------------------------------------------
# REQ-044 — Agent C and Agent D outputs are byte-for-byte deterministic.
# Re-running must NOT change finding timestamps (no wall-clock now()).
# ---------------------------------------------------------------------------

def test_coverage_and_settlement_are_deterministic(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _run_through_c(runs_dir, "scenario_02_total_loss", "CLM-2026-002")
    run_dir = runs_dir / "CLM-2026-002"

    # Establish the baseline: C already ran, run D once.
    _run_damage(runs_dir, "CLM-2026-002")
    coverage_first = (run_dir / "coverage_result.json").read_bytes()
    settlement_first = (run_dir / "settlement_calc.json").read_bytes()

    # Re-run C and D against the same inputs.
    claim_summary = _load(run_dir / "claim_summary.json")
    policy = _load(run_dir / "context_packet.json")["policy"]
    AgentCoverageValidation(runs_dir=runs_dir).process(claim_summary, policy)
    _run_damage(runs_dir, "CLM-2026-002")

    assert (run_dir / "coverage_result.json").read_bytes() == coverage_first
    assert (run_dir / "settlement_calc.json").read_bytes() == settlement_first
