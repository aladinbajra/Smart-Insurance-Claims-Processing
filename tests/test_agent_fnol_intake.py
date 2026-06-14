from __future__ import annotations

from pathlib import Path

import pytest

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


def _agent(tmp_path: Path) -> AgentFNOLIntake:
    root = Path(__file__).resolve().parents[1]
    return AgentFNOLIntake(
        config_path=root / "config" / "policy.yaml",
        runs_dir=tmp_path / "runs",
    )


def _finding_codes(result: dict) -> list[str]:
    """Return the short code portion of every finding_id (e.g. 'NEW_CLAIMANT')."""
    return [f["finding_id"].split("-", 3)[-1] for f in result["findings"]]


def _artifacts_exist(result: dict, runs_root: Path) -> bool:
    run_dir = runs_root / result["claim_id"]
    return all(
        (run_dir / name).exists()
        for name in (
            "context_packet.json",
            "evidence_index.json",
            "audit_log.md",
        )
    )


# ---------------------------------------------------------------------------
# Scenario 01 — Clean auto claim  (straight-through, no findings)
# ---------------------------------------------------------------------------

def test_scenario_01_clean_auto(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = agent.process(_dataset("scenario_01_clean_auto"))

    assert result["claim_id"] == "CLM-2026-001"
    assert result["claim_type"] == "auto"
    assert result["status"] == "ready_for_document_extraction"
    assert result["next_agent"] == "agent_b"
    assert result["findings"] == []

    # 7 evidence entries: fnol + policy + repair_estimate + police_report
    #                     + vendor_master + 2 reference-only photos
    assert len(result["evidence_index"]) == 7

    photos = [e for e in result["evidence_index"] if e["doc_type"] == "photo"]
    assert len(photos) == 2
    assert all(not e["processable"] for e in photos)

    assert _artifacts_exist(result, tmp_path / "runs")


# ---------------------------------------------------------------------------
# Scenario 02 — Total loss (high repair vs market value, no intake risk)
# ---------------------------------------------------------------------------

def test_scenario_02_total_loss(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = agent.process(_dataset("scenario_02_total_loss"))

    assert result["claim_id"] == "CLM-2026-002"
    assert result["claim_type"] == "auto"

    # No adverse intake signals — total-loss detection is Agent D's job
    assert result["status"] == "ready_for_document_extraction"
    assert result["next_agent"] == "agent_b"
    assert result["findings"] == []

    # 8 entries: fnol + policy + estimate + police_report + vendor_master
    #            + 2 photos + 1 medical_report (all reference-only non-PDF)
    assert len(result["evidence_index"]) == 8

    assert _artifacts_exist(result, tmp_path / "runs")


# ---------------------------------------------------------------------------
# Scenario 03 — SIU referral (fraud history + excessive prior claims)
# ---------------------------------------------------------------------------

def test_scenario_03_siu_referral(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = agent.process(_dataset("scenario_03_siu_referral"))

    assert result["claim_id"] == "CLM-2026-003"

    # High-severity findings must isolate the claim at intake
    assert result["status"] == "intake_review_required"
    assert result["next_agent"] is None

    codes = _finding_codes(result)
    assert "PRIOR_FRAUD_HISTORY" in codes      # fraud_history: true  → high
    assert "MULTIPLE_PRIOR_CLAIMS" in codes    # 4 claims ≥ threshold → medium

    # At least one high-severity finding must be present
    severities = [f["severity"] for f in result["findings"]]
    assert "high" in severities

    # 6 entries: fnol + policy + estimate + vendor_master + 2 photos
    # (no police_report — none was filed)
    assert len(result["evidence_index"]) == 6

    assert _artifacts_exist(result, tmp_path / "runs")


# ---------------------------------------------------------------------------
# Scenario 04 — Coverage denial (new claimant, incident pre-dates inception)
# ---------------------------------------------------------------------------

def test_scenario_04_coverage_denial(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = agent.process(_dataset("scenario_04_coverage_denial"))

    assert result["claim_id"] == "CLM-2026-004"

    # new_claimant triggers a medium finding — processing continues with warning
    assert result["status"] == "ready_for_document_extraction_with_warnings"
    assert result["next_agent"] == "agent_b"

    codes = _finding_codes(result)
    assert "NEW_CLAIMANT" in codes     # medium — does not isolate

    # Coverage denial is Agent C's responsibility; Agent A must not pre-empt it
    assert "INACTIVE_OR_EXPIRED_POLICY" not in codes

    # 6 entries: fnol + policy + estimate + vendor_master + 2 photos
    assert len(result["evidence_index"]) == 6

    assert _artifacts_exist(result, tmp_path / "runs")


# ---------------------------------------------------------------------------
# Scenario 05 — Late notice (91 days; late-notice check belongs to Agent C)
# ---------------------------------------------------------------------------

def test_scenario_05_late_notice(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = agent.process(_dataset("scenario_05_late_notice"))

    assert result["claim_id"] == "CLM-2026-005"
    assert result["claim_type"] == "auto"

    # Agent A has no late-notice rule — claim passes intake cleanly
    assert result["status"] == "ready_for_document_extraction"
    assert result["next_agent"] == "agent_b"
    assert result["findings"] == []

    # 7 entries: fnol + policy + estimate + police_report + vendor_master + 2 photos
    assert len(result["evidence_index"]) == 7

    assert _artifacts_exist(result, tmp_path / "runs")


# ---------------------------------------------------------------------------
# Scenario 06 — CAT event (hailstorm, Naperville ZIP matches active event)
# ---------------------------------------------------------------------------

def test_scenario_06_cat_event(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = agent.process(_dataset("scenario_06_cat_event"))

    assert result["claim_id"] == "CLM-2026-006"

    # CAT flag is medium — processing continues with warning
    assert result["status"] == "ready_for_document_extraction_with_warnings"
    assert result["next_agent"] == "agent_b"

    codes = _finding_codes(result)
    assert "CATASTROPHE_EVENT" in codes    # incident.cat_event: true → medium

    # 7 entries: fnol + policy + estimate + police_report + vendor_master + 2 photos
    assert len(result["evidence_index"]) == 7

    assert _artifacts_exist(result, tmp_path / "runs")


# ---------------------------------------------------------------------------
# Scenario 07 — Fraud ring (blacklisted provider PROV-999, RING-CHI-001)
# ---------------------------------------------------------------------------

def test_scenario_07_fraud_ring(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = agent.process(_dataset("scenario_07_fraud_ring"))

    assert result["claim_id"] == "CLM-2026-007"

    # Multiple high findings — claim must be isolated immediately
    assert result["status"] == "intake_review_required"
    assert result["next_agent"] is None

    codes = _finding_codes(result)
    assert "PRIOR_FRAUD_HISTORY" in codes       # fraud_history: true → high
    assert "BLACKLISTED_PROVIDER" in codes      # PROV-999 → high
    assert "KNOWN_FRAUD_RING_MATCH" in codes    # RING-CHI-001 → high

    # All three signals must require human review
    assert all(f["requires_human_review"] for f in result["findings"]
               if f["finding_id"].split("-", 3)[-1] in
               {"PRIOR_FRAUD_HISTORY", "BLACKLISTED_PROVIDER", "KNOWN_FRAUD_RING_MATCH"})

    # 8 entries: fnol + policy + estimate + police_report + vendor_master
    #            + 2 photos + 1 medical_report (reference-only)
    assert len(result["evidence_index"]) == 8

    assert _artifacts_exist(result, tmp_path / "runs")


# ---------------------------------------------------------------------------
# Scenario 08 — Low OCR (handwritten estimate; OCR quality is Agent B's concern)
# ---------------------------------------------------------------------------

def test_scenario_08_low_ocr(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = agent.process(_dataset("scenario_08_low_ocr"))

    assert result["claim_id"] == "CLM-2026-008"
    assert result["claim_type"] == "auto"

    # OCR confidence is not an Agent A signal — claim passes intake cleanly
    assert result["status"] == "ready_for_document_extraction"
    assert result["next_agent"] == "agent_b"
    assert result["findings"] == []

    # 6 entries: fnol + policy + estimate + vendor_master + 2 photos
    assert len(result["evidence_index"]) == 6

    assert _artifacts_exist(result, tmp_path / "runs")


# ---------------------------------------------------------------------------
# Scenario 09 — Homeowners clean claim (different claim type, no vehicle)
# ---------------------------------------------------------------------------

def test_scenario_09_homeowners(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = agent.process(_dataset("scenario_09_homeowners"))

    assert result["claim_id"] == "CLM-2026-009"
    assert result["claim_type"] == "homeowners"

    assert result["status"] == "ready_for_document_extraction"
    assert result["next_agent"] == "agent_b"
    assert result["findings"] == []

    # 4 entries only: fnol + policy + estimate + vendor_master (no photos)
    assert len(result["evidence_index"]) == 4

    assert _artifacts_exist(result, tmp_path / "runs")


# ---------------------------------------------------------------------------
# Scenario 10 — Expired policy (policy_expired: true at time of incident)
# ---------------------------------------------------------------------------

def test_scenario_10_expired_policy(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = agent.process(_dataset("scenario_10_expired_policy"))

    assert result["claim_id"] == "CLM-2026-010"

    # Expired policy is a high-severity intake signal — isolate immediately
    assert result["status"] == "intake_review_required"
    assert result["next_agent"] is None

    codes = _finding_codes(result)
    assert "INACTIVE_OR_EXPIRED_POLICY" in codes    # policy_expired: true → high

    high_findings = [f for f in result["findings"] if f["severity"] == "high"]
    assert len(high_findings) >= 1

    # 7 entries: fnol + policy + estimate + police_report + vendor_master + 2 photos
    assert len(result["evidence_index"]) == 7

    assert _artifacts_exist(result, tmp_path / "runs")


# ---------------------------------------------------------------------------
# Scenario 11 — Missing documents (repair estimate not submitted)
# ---------------------------------------------------------------------------

def test_scenario_11_missing_documents(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = agent.process(_dataset("scenario_11_missing_documents"))

    assert result["claim_id"] == "CLM-2026-011"

    # Missing documents flag is high — isolate at intake
    assert result["status"] == "intake_review_required"
    assert result["next_agent"] is None

    codes = _finding_codes(result)
    assert "MISSING_DOCUMENTS_FLAG" in codes    # flags.missing_documents: true → high

    # 5 entries: fnol + policy + estimate(reference-only) + police_report + vendor_master
    assert len(result["evidence_index"]) == 5

    assert _artifacts_exist(result, tmp_path / "runs")


# ---------------------------------------------------------------------------
# Scenario 12 — Duplicate claim (same VIN + incident as CLM-2026-007,
#                                 blacklisted provider PROV-999)
# ---------------------------------------------------------------------------

def test_scenario_12_duplicate_claim(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = agent.process(_dataset("scenario_12_duplicate_claim"))

    assert result["claim_id"] == "CLM-2026-012"

    # Duplicate + blacklisted provider + fraud ring = multiple high findings
    assert result["status"] == "intake_review_required"
    assert result["next_agent"] is None

    codes = _finding_codes(result)
    assert "POSSIBLE_DUPLICATE_CLAIM" in codes    # flags.duplicate_claim: true → high
    assert "NEW_CLAIMANT" in codes                # new_claimant: true → medium
    assert "BLACKLISTED_PROVIDER" in codes        # PROV-999 → high
    assert "KNOWN_FRAUD_RING_MATCH" in codes      # RING-CHI-001 → high

    # 4 findings total
    assert len(result["findings"]) == 4

    # 8 entries: fnol + policy + estimate + police_report + vendor_master
    #            + 2 photos + 1 medical_report (reference-only)
    assert len(result["evidence_index"]) == 8

    assert _artifacts_exist(result, tmp_path / "runs")


# ---------------------------------------------------------------------------
# Edge case — Missing manifest raises FileNotFoundError
# ---------------------------------------------------------------------------

def test_missing_manifest_raises(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    empty_bundle = tmp_path / "empty_bundle"
    empty_bundle.mkdir()

    with pytest.raises(FileNotFoundError, match="manifest"):
        agent.process(empty_bundle)
