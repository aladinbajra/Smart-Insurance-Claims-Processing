from __future__ import annotations

import json
from pathlib import Path

from backend.agents.agent_coverage_validation import AgentCoverageValidation
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


def _run_a_then_b(runs_dir: Path, scenario: str) -> None:
    """Produce context_packet.json + claim_summary.json on disk."""
    AgentFNOLIntake(config_path=_config(), runs_dir=runs_dir).process(
        _dataset(scenario)
    )
    AgentDocumentExtraction(config_path=_config(), runs_dir=runs_dir).process(
        _dataset(scenario)
    )


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _run_coverage(runs_dir: Path, claim_id: str) -> dict:
    """Run Agent C from the on-disk artifacts and return its result dict."""
    run_dir = runs_dir / claim_id
    claim_summary = _load(run_dir / "claim_summary.json")
    context = _load(run_dir / "context_packet.json")
    policy = context["policy"]

    agent_c = AgentCoverageValidation(runs_dir=runs_dir)
    result = agent_c.process(claim_summary, policy)
    return result.model_dump(mode="json")


def _finding_codes(result: dict) -> list[str]:
    return [f["finding_id"] for f in result["findings"]]


# ---------------------------------------------------------------------------
# Scenario 01 — clean auto: coverage active, no denial, result persisted
# ---------------------------------------------------------------------------

def test_scenario_01_coverage_active(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _run_a_then_b(runs_dir, "scenario_01_clean_auto")
    result = _run_coverage(runs_dir, "CLM-2026-001")

    assert result["claim_id"] == "CLM-2026-001"
    assert result["coverage_active"] is True
    assert result["policy_expired"] is False
    assert result["denial_reason"] is None
    assert result["late_notice_violation"] is False
    assert result["exclusions_triggered"] == []
    assert result["deductible"] == 1000.0
    assert result["coverage_limit"] == 100000.0
    assert result["applicable_state"] == "IL"

    # BUG-C1: coverage_result.json must be written to the run folder.
    assert (runs_dir / "CLM-2026-001" / "coverage_result.json").exists()


# ---------------------------------------------------------------------------
# Scenario 10 — expired policy: denial derived from dates (BUG-C2)
# ---------------------------------------------------------------------------

def test_scenario_10_policy_expired(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _run_a_then_b(runs_dir, "scenario_10_expired_policy")
    result = _run_coverage(runs_dir, "CLM-2026-010")

    assert result["coverage_active"] is False
    assert result["policy_expired"] is True
    assert result["denial_reason"] == "Policy expired"
    assert "C-001-POLICY_EXPIRED" in _finding_codes(result)


# ---------------------------------------------------------------------------
# Scenario 04 — pre-inception loss: denial derived from dates (BUG-C2)
# ---------------------------------------------------------------------------

def test_scenario_04_pre_inception(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _run_a_then_b(runs_dir, "scenario_04_coverage_denial")
    result = _run_coverage(runs_dir, "CLM-2026-004")

    assert result["coverage_active"] is False
    assert result["policy_expired"] is False
    assert result["denial_reason"] == "Pre-inception loss event"
    assert "C-002-COVERAGE_DENIED" in _finding_codes(result)


# ---------------------------------------------------------------------------
# Scenario 05 — late notice: coverage stays active, violation flagged
# ---------------------------------------------------------------------------

def test_scenario_05_late_notice(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _run_a_then_b(runs_dir, "scenario_05_late_notice")
    result = _run_coverage(runs_dir, "CLM-2026-005")

    assert result["coverage_active"] is True
    assert result["late_notice_violation"] is True
    assert result["days_to_report"] == 91
    assert result["late_notice_days_allowed"] == 30
    assert "C-003-LATE_NOTICE" in _finding_codes(result)


# ---------------------------------------------------------------------------
# BUG-X1 regression — Agent C must consume claim_summary.json, not the
# manifest ground truth. Editing the summary on disk must change the outcome.
# ---------------------------------------------------------------------------

def test_consumes_claim_summary_not_manifest(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _run_a_then_b(runs_dir, "scenario_01_clean_auto")

    summary_path = runs_dir / "CLM-2026-001" / "claim_summary.json"
    summary = _load(summary_path)

    # Manifest says the policy is NOT expired. Tamper with Agent B's output:
    # push the incident date well past the policy expiration (2027-03-01).
    summary["incident_date"] = "2030-01-01"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh)

    context = _load(runs_dir / "CLM-2026-001" / "context_packet.json")
    result = AgentCoverageValidation(runs_dir=runs_dir).process(
        summary, context["policy"]
    )

    # If C read the manifest it would report active coverage; reading the
    # tampered summary it must now compute an expiry-based denial.
    assert result.policy_expired is True
    assert result.coverage_active is False
    assert result.denial_reason == "Policy expired"
