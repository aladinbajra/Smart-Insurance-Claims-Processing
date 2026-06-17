from __future__ import annotations

import json
from pathlib import Path

import yaml

from backend.agents.agent_coverage_validation import AgentCoverageValidation
from backend.agents.agent_damage_assessment import AgentDamageAssessment
from backend.agents.agent_document_extraction import AgentDocumentExtraction
from backend.agents.agent_fnol_intake import AgentFNOLIntake
from backend.agents.agent_fraud_detection import AgentFraudDetection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dataset(scenario: str) -> Path:
    return _root() / "datasets" / "SICPS_Dataset (3)" / "SICPS_Dataset" / scenario


def _config() -> Path:
    return _root() / "config" / "policy.yaml"


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _run_a(runs_dir: Path, scenario: str, claim_id: str) -> Path:
    """Run Agent A only — Agent E's required input is context_packet.json."""
    AgentFNOLIntake(config_path=_config(), runs_dir=runs_dir).process(_dataset(scenario))
    return runs_dir / claim_id


def _run_full_to_d(runs_dir: Path, scenario: str, claim_id: str) -> Path:
    """Run A -> B -> C -> D so Agent E can use its optional inputs too."""
    cfg = _config()
    AgentFNOLIntake(config_path=cfg, runs_dir=runs_dir).process(_dataset(scenario))
    AgentDocumentExtraction(config_path=cfg, runs_dir=runs_dir).process(_dataset(scenario))
    run_dir = runs_dir / claim_id
    claim_summary = _load(run_dir / "claim_summary.json")
    policy = _load(run_dir / "context_packet.json")["policy"]
    AgentCoverageValidation(runs_dir=runs_dir).process(claim_summary, policy)
    coverage = _load(run_dir / "coverage_result.json")
    AgentDamageAssessment(config_path=cfg, runs_dir=runs_dir).process(claim_summary, coverage)
    return run_dir


def _indicator_names(result: dict) -> list[str]:
    return [i["indicator"] for i in result["indicators"]]


# ---------------------------------------------------------------------------
# Scenario 01 — clean auto: zero fraud signals
# ---------------------------------------------------------------------------

def test_scenario_01_clean_low_risk(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = _run_a(runs_dir, "scenario_01_clean_auto", "CLM-2026-001")
    result = AgentFraudDetection(config_path=_config()).process(run_dir)

    assert result["fraud_score"] == 0.0
    assert result["risk_level"] == "low"
    assert result["siu_referral"] is False
    assert result["adjuster_review"] is False
    assert result["indicators"] == []
    assert (run_dir / "fraud_score.json").exists()


# ---------------------------------------------------------------------------
# Scenario 03 — SIU via claim frequency + fraud history (REQ-021, REQ-025)
# ---------------------------------------------------------------------------

def test_scenario_03_claim_frequency_siu(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = _run_a(runs_dir, "scenario_03_siu_referral", "CLM-2026-003")
    result = AgentFraudDetection(config_path=_config()).process(run_dir)

    assert result["prior_claims_trigger"] is True          # 4 >= 3
    assert result["siu_referral"] is True
    assert result["risk_level"] in ("high", "critical")
    names = _indicator_names(result)
    assert "multiple_prior_claims" in names
    assert "prior_fraud_history" in names


# ---------------------------------------------------------------------------
# Scenario 07 — known fraud ring + blacklisted provider (REQ-024, REQ-025)
# ---------------------------------------------------------------------------

def test_scenario_07_fraud_ring_blacklist(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = _run_a(runs_dir, "scenario_07_fraud_ring", "CLM-2026-007")
    result = AgentFraudDetection(config_path=_config()).process(run_dir)

    assert result["provider_blacklisted"] is True
    assert result["provider_ring_match"] is True
    assert result["provider_ring_id"] == "RING-CHI-001"
    assert result["siu_referral"] is True
    assert result["risk_level"] == "critical"
    names = _indicator_names(result)
    assert "blacklisted_provider" in names
    assert "known_fraud_ring_match" in names


# ---------------------------------------------------------------------------
# REQ-044 — deterministic output: identical inputs -> byte-identical file
# ---------------------------------------------------------------------------

def test_fraud_score_is_deterministic(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = _run_a(runs_dir, "scenario_07_fraud_ring", "CLM-2026-007")
    agent = AgentFraudDetection(config_path=_config())

    agent.process(run_dir)
    first = (run_dir / "fraud_score.json").read_bytes()
    agent.process(run_dir)
    second = (run_dir / "fraud_score.json").read_bytes()

    assert first == second


# ---------------------------------------------------------------------------
# REQ-023 — thresholds are read from policy.yaml (configurability)
# Raising prior_claims_siu_trigger above the claim count flips the trigger.
# ---------------------------------------------------------------------------

def test_prior_claims_trigger_is_configurable(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = _run_a(runs_dir, "scenario_03_siu_referral", "CLM-2026-003")

    baseline = AgentFraudDetection(config_path=_config()).process(run_dir)
    assert baseline["prior_claims_trigger"] is True   # default trigger = 3, claim has 4

    with _config().open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["fraud"]["prior_claims_siu_trigger"] = 5       # now 4 < 5
    custom = tmp_path / "policy_high_trigger.yaml"
    with custom.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)

    adjusted = AgentFraudDetection(config_path=custom).process(run_dir)
    assert adjusted["prior_claims_trigger"] is False
    assert "multiple_prior_claims" not in _indicator_names(adjusted)


# ---------------------------------------------------------------------------
# Optional-input integration — Agent E still runs after A->B->C->D
# ---------------------------------------------------------------------------

def test_runs_with_full_upstream_pipeline(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = _run_full_to_d(runs_dir, "scenario_07_fraud_ring", "CLM-2026-007")
    result = AgentFraudDetection(config_path=_config()).process(run_dir)

    # Still flags the ring even with coverage_result/settlement_calc present.
    assert result["provider_ring_match"] is True
    assert result["siu_referral"] is True
    assert (run_dir / "fraud_score.json").exists()
