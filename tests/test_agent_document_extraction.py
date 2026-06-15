from __future__ import annotations

from pathlib import Path

import pytest

from backend.agents.agent_document_extraction import AgentDocumentExtraction
from backend.agents.agent_fnol_intake import AgentFNOLIntake


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dataset(scenario: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    return (
        root
        / "datasets"
        / "SICPS_Dataset (3)"
        / "SICPS_Dataset"
        / scenario
    )


def _run_a_then_b(tmp_path: Path, scenario: str) -> dict:
    """Agent B requires Agent A's context_packet — run the pair in order."""
    root = Path(__file__).resolve().parents[1]
    config = root / "config" / "policy.yaml"
    runs_dir = tmp_path / "runs"

    AgentFNOLIntake(config_path=config, runs_dir=runs_dir).process(
        _dataset(scenario)
    )
    agent_b = AgentDocumentExtraction(config_path=config, runs_dir=runs_dir)
    return agent_b.process(_dataset(scenario))


def _finding_codes(result: dict) -> list[str]:
    return [f["finding_id"].split("-", 2)[-1] for f in result["findings"]]


# ---------------------------------------------------------------------------
# Scenario 01 — Clean auto: high confidence, no manual review
# ---------------------------------------------------------------------------

def test_scenario_01_clean_extraction(tmp_path: Path) -> None:
    result = _run_a_then_b(tmp_path, "scenario_01_clean_auto")

    assert result["claim_id"] == "CLM-2026-001"
    assert result["claim_type"] == "auto"
    assert result["overall_confidence"] >= 0.70
    assert result["requires_manual_review"] is False
    assert result["extraction_errors"] == []
    assert result["status"] == "ready_for_coverage_validation"
    assert result["next_agent"] == "agent_c"

    # Every extracted field must carry a confidence score.
    assert len(result["extracted_fields"]) > 0
    assert all("confidence" in f for f in result["extracted_fields"])

    # claim_summary.json must be written to the run folder.
    assert (tmp_path / "runs" / "CLM-2026-001" / "claim_summary.json").exists()


# ---------------------------------------------------------------------------
# Scenario 08 — Low OCR: confidence below threshold, manual review required
# ---------------------------------------------------------------------------

def test_scenario_08_low_ocr_requires_review(tmp_path: Path) -> None:
    result = _run_a_then_b(tmp_path, "scenario_08_low_ocr")

    assert result["claim_id"] == "CLM-2026-008"
    assert result["overall_confidence"] < 0.70
    assert result["requires_manual_review"] is True
    assert result["status"] == "ready_for_coverage_validation_with_review"

    # Many fields below the flag threshold + a LOW_OCR_CONFIDENCE finding.
    assert len(result["low_confidence_fields"]) > 0
    assert "LOW_OCR_CONFIDENCE" in _finding_codes(result)


# ---------------------------------------------------------------------------
# Scenario 11 — Missing required document (repair_estimate not submitted)
# ---------------------------------------------------------------------------

def test_scenario_11_missing_required_document(tmp_path: Path) -> None:
    result = _run_a_then_b(tmp_path, "scenario_11_missing_documents")

    assert result["claim_id"] == "CLM-2026-011"
    assert result["requires_manual_review"] is True
    assert len(result["extraction_errors"]) >= 1
    assert any("repair_estimate" in e for e in result["extraction_errors"])
    assert "MISSING_REQUIRED_DOCUMENT" in _finding_codes(result)


# ---------------------------------------------------------------------------
# Scenario 09 — Homeowners: no vehicle, still high confidence
# ---------------------------------------------------------------------------

def test_scenario_09_homeowners_no_vehicle(tmp_path: Path) -> None:
    result = _run_a_then_b(tmp_path, "scenario_09_homeowners")

    assert result["claim_id"] == "CLM-2026-009"
    assert result["claim_type"] == "homeowners"
    assert result["requires_manual_review"] is False

    # Homeowners claims have no vehicle — VIN must be absent, market value 0.
    assert result["vehicle_vin"] is None
    assert result["market_value"] == 0.0
    # Denormalized financials resolve from the bundle.
    assert result["repair_estimate"] == 3220.0
    assert result["gross_amount"] == 3220.0


# ---------------------------------------------------------------------------
# Scenario 07 — Fraud ring: extraction succeeds, denormalized provider present
# ---------------------------------------------------------------------------

def test_scenario_07_denormalized_provider(tmp_path: Path) -> None:
    result = _run_a_then_b(tmp_path, "scenario_07_fraud_ring")

    assert result["claim_id"] == "CLM-2026-007"
    assert result["requires_manual_review"] is False  # documents are clean quality

    # Provider identifiers must be available for Agent E fraud scoring.
    assert result["provider_id"] == "PROV-999"
    assert result["provider_tax_id"] == "88-1234567"
    assert result["repair_estimate"] == 26300.0
    assert result["medical_total"] == 9800.0
    assert result["gross_amount"] == 36100.0


# ---------------------------------------------------------------------------
# Scenario 02 — Total loss: market value + repair estimate denormalized
# ---------------------------------------------------------------------------

def test_scenario_02_total_loss_values(tmp_path: Path) -> None:
    result = _run_a_then_b(tmp_path, "scenario_02_total_loss")

    assert result["claim_id"] == "CLM-2026-002"
    assert result["repair_estimate"] == 34600.0
    assert result["market_value"] == 32000.0
    assert result["vehicle_vin"] == "1FTEW1EP5LFA12345"


# ---------------------------------------------------------------------------
# Pipeline order — Agent B without Agent A must fail clearly
# ---------------------------------------------------------------------------

def test_requires_agent_a_first(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    agent_b = AgentDocumentExtraction(
        config_path=root / "config" / "policy.yaml",
        runs_dir=tmp_path / "runs",
    )
    with pytest.raises(FileNotFoundError, match="context_packet"):
        agent_b.process(_dataset("scenario_01_clean_auto"))
