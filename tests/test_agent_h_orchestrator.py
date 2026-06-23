from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.agent_coverage_validation import AgentCoverageValidation
from backend.agents.agent_damage_assessment import AgentDamageAssessment
from backend.agents.agent_document_extraction import AgentDocumentExtraction
from backend.agents.agent_exception_triage import AgentExceptionTriage
from backend.agents.agent_fnol_intake import AgentFNOLIntake
from backend.agents.agent_fraud_detection import AgentFraudDetection


# ---------------------------------------------------------------------------
# Helpers — run the full pipeline A -> B -> C -> D -> E -> H
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


def _run_full_pipeline(runs_dir: Path, scenario: str, claim_id: str) -> Path:
    cfg = _config()
    AgentFNOLIntake(config_path=cfg, runs_dir=runs_dir).process(_dataset(scenario))
    AgentDocumentExtraction(config_path=cfg, runs_dir=runs_dir).process(_dataset(scenario))

    run_dir = runs_dir / claim_id
    claim_summary = _load(run_dir / "claim_summary.json")
    policy = _load(run_dir / "context_packet.json")["policy"]

    AgentCoverageValidation(runs_dir=runs_dir).process(claim_summary, policy)
    coverage = _load(run_dir / "coverage_result.json")
    AgentDamageAssessment(config_path=cfg, runs_dir=runs_dir).process(claim_summary, coverage)
    AgentFraudDetection(config_path=cfg).process(run_dir)
    AgentExceptionTriage(config_path=cfg).process(run_dir)
    return run_dir


# Authoritative expected routing per scenario (manifest "denial" == enum
# "coverage_denial"). Source: dataset expected_agent_signals.agent_exception_triage.
_EXPECTED = [
    ("scenario_01_clean_auto", "CLM-2026-001", "auto_settle"),
    # Total loss, late notice, low-OCR/missing docs and CAT events each get a
    # distinct routing category (not a generic adjuster_review bucket).
    ("scenario_02_total_loss", "CLM-2026-002", "total_loss_routing"),
    ("scenario_03_siu_referral", "CLM-2026-003", "siu_referral"),
    ("scenario_04_coverage_denial", "CLM-2026-004", "coverage_denial"),
    ("scenario_05_late_notice", "CLM-2026-005", "late_notice_review"),
    ("scenario_06_cat_event", "CLM-2026-006", "cat_surge_processing"),
    ("scenario_07_fraud_ring", "CLM-2026-007", "siu_referral"),
    ("scenario_08_low_ocr", "CLM-2026-008", "manual_review_route"),
    ("scenario_09_homeowners", "CLM-2026-009", "auto_settle"),
    ("scenario_10_expired_policy", "CLM-2026-010", "coverage_denial"),
    ("scenario_11_missing_documents", "CLM-2026-011", "manual_review_route"),
    # Scenario 12 uses provider PROV-999 (blacklisted + RING-CHI-001), so the
    # single-claim pipeline correctly escalates to SIU. The dataset labels it
    # adjuster_review only because it knows the claim duplicates the already-
    # referred CLM-2026-007 — cross-claim dedup is out of scope (stretch goal).
    ("scenario_12_duplicate_claim", "CLM-2026-012", "siu_referral"),
]


@pytest.mark.parametrize("scenario,claim_id,expected_routing", _EXPECTED)
def test_routing_matches_ground_truth(
    tmp_path: Path, scenario: str, claim_id: str, expected_routing: str
) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = _run_full_pipeline(runs_dir, scenario, claim_id)
    approval = _load(run_dir / "approval_packet.json")

    assert approval["routing_decision"] == expected_routing
    # SIU and denial never pay out.
    if expected_routing in ("siu_referral", "coverage_denial"):
        assert approval["net_settlement"] == 0.0


# ---------------------------------------------------------------------------
# Artifact production (REQ-029)
# ---------------------------------------------------------------------------

def test_all_required_artifacts_written(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = _run_full_pipeline(runs_dir, "scenario_01_clean_auto", "CLM-2026-001")

    for artifact in (
        "approval_packet.json",
        "exceptions.md",
        "audit_log.md",
        "metrics.json",
    ):
        assert (run_dir / artifact).exists(), artifact

    approval = _load(run_dir / "approval_packet.json")
    assert approval["claim_id"] == "CLM-2026-001"
    assert approval["routing_decision"] == "auto_settle"
    assert approval["net_settlement"] == 550.0
    assert approval["adjuster_notes"]  # non-empty (threaded from manifest)
    assert approval["exceptions"] == []  # clean claim


# ---------------------------------------------------------------------------
# Clean claim -> settlement finalized, metrics populated
# ---------------------------------------------------------------------------

def test_settlement_finalized_and_metrics(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = _run_full_pipeline(runs_dir, "scenario_09_homeowners", "CLM-2026-009")

    settlement = _load(run_dir / "settlement_calc.json")
    assert settlement["finalized"] is True

    metrics = _load(run_dir / "metrics.json")
    assert metrics["routing_decision"] == "auto_settle"
    assert metrics["agents_executed"] == [
        "agent_fnol_intake",
        "agent_document_extraction",
        "agent_coverage_validation",
        "agent_damage_assessment",
        "agent_fraud_detection",
        "agent_exception_triage",
    ]
    assert metrics["total_findings"] == sum(metrics["findings_by_severity"].values())
    assert metrics["extraction_field_count"] > 0


# ---------------------------------------------------------------------------
# Fraud ring -> SIU exception with ring evidence, zero payout
# ---------------------------------------------------------------------------

def test_fraud_ring_siu_exception(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = _run_full_pipeline(runs_dir, "scenario_07_fraud_ring", "CLM-2026-007")
    approval = _load(run_dir / "approval_packet.json")

    assert approval["routing_decision"] == "siu_referral"
    assert approval["net_settlement"] == 0.0
    categories = [e["category"] for e in approval["exceptions"]]
    assert "fraud_siu" in categories
    siu_exc = next(e for e in approval["exceptions"] if e["category"] == "fraud_siu")
    assert siu_exc["assigned_to"] == "SIU"


# ---------------------------------------------------------------------------
# Coverage denial -> denial exception, zero payout
# ---------------------------------------------------------------------------

def test_coverage_denial_exception(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = _run_full_pipeline(runs_dir, "scenario_10_expired_policy", "CLM-2026-010")
    approval = _load(run_dir / "approval_packet.json")

    assert approval["routing_decision"] == "coverage_denial"
    assert approval["net_settlement"] == 0.0
    assert "coverage_denial" in [e["category"] for e in approval["exceptions"]]


# ---------------------------------------------------------------------------
# REQ-028 / REQ-044 — idempotent, byte-for-byte deterministic outputs
# ---------------------------------------------------------------------------

def test_agent_h_is_idempotent(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = _run_full_pipeline(runs_dir, "scenario_07_fraud_ring", "CLM-2026-007")

    approval_first = (run_dir / "approval_packet.json").read_bytes()
    metrics_first = (run_dir / "metrics.json").read_bytes()
    audit_first = (run_dir / "audit_log.md").read_bytes()
    exceptions_first = (run_dir / "exceptions.md").read_bytes()

    # Re-run Agent H alone on the same run folder.
    AgentExceptionTriage(config_path=_config()).process(run_dir)

    assert (run_dir / "approval_packet.json").read_bytes() == approval_first
    assert (run_dir / "metrics.json").read_bytes() == metrics_first
    assert (run_dir / "audit_log.md").read_bytes() == audit_first
    assert (run_dir / "exceptions.md").read_bytes() == exceptions_first


# ---------------------------------------------------------------------------
# Findings are merged across all upstream agents
# ---------------------------------------------------------------------------

def test_findings_merged_from_all_agents(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = _run_full_pipeline(runs_dir, "scenario_03_siu_referral", "CLM-2026-003")
    approval = _load(run_dir / "approval_packet.json")

    agents = {f["agent"] for f in approval["all_findings"]}
    # Agent A (intake) and Agent E (fraud) both raise findings for this claim.
    assert "agent_fnol_intake" in agents
    assert "agent_fraud_detection" in agents
    assert len(approval["evidence_bundle"]) > 0


# ---------------------------------------------------------------------------
# Missing context_packet -> clear error
# ---------------------------------------------------------------------------

def test_requires_context_packet(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="context_packet"):
        AgentExceptionTriage(config_path=_config()).process(tmp_path / "runs" / "NOPE")


# ---------------------------------------------------------------------------
# REQ-051 — tracker work-item (ADO/Jira stub), idempotent per claim
# ---------------------------------------------------------------------------

def test_tracker_ticket_for_siu(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = _run_full_pipeline(runs_dir, "scenario_07_fraud_ring", "CLM-2026-007")
    ticket = _load(run_dir / "tracker_ticket.json")

    assert ticket["ticket_id"] == "SICPS-CLM-2026-007"  # stable id (idempotent)
    assert ticket["status"] == "open"
    assert ticket["assigned_to"] == "SIU"
    assert ticket["priority"] == "critical"
    assert ticket["routing_decision"] == "siu_referral"


def test_tracker_ticket_auto_settle_is_closed(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = _run_full_pipeline(runs_dir, "scenario_01_clean_auto", "CLM-2026-001")
    ticket = _load(run_dir / "tracker_ticket.json")

    assert ticket["status"] == "closed"
    assert ticket["assigned_to"] == "auto"


# ---------------------------------------------------------------------------
# REQ-029 — metrics include an exception rate
# ---------------------------------------------------------------------------

def test_metrics_exception_rate(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = _run_full_pipeline(runs_dir, "scenario_01_clean_auto", "CLM-2026-001")
    metrics = _load(run_dir / "metrics.json")

    assert "exception_rate" in metrics
    assert metrics["exception_rate"] == 0.0  # clean claim -> no exceptions


# ---------------------------------------------------------------------------
# Stretch agents now CONTRIBUTE: non-compliance and genuine anomalies become
# exceptions in the approval packet (REQ-048 / REQ-052).
# ---------------------------------------------------------------------------

def _write(run_dir: Path, name: str, data: dict) -> None:
    with (run_dir / name).open("w", encoding="utf-8") as fh:
        json.dump(data, fh)


def test_compliance_and_anomaly_raise_exceptions(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "CLM-SYNTH-001"
    run_dir.mkdir(parents=True)
    _write(run_dir, "context_packet.json", {
        "claim_id": "CLM-SYNTH-001", "run_id": "CLM-SYNTH-001",
        "created_at": "1970-01-01T00:00:00Z", "claim_type": "auto",
        "flags": {}, "incident": {}, "input_hash": "x",
    })
    _write(run_dir, "compliance_result.json", {
        "compliant": False, "framework": "NAIC",
        "checks": [{"check": "pii_redaction_enabled", "passed": False}],
        "findings": [],
    })
    _write(run_dir, "anomaly_report.json", {
        "anomaly_score": 0.25,
        "anomalies": [{"code": "HIGH_VALUE_OUTLIER", "detail": "gross too high"}],
        "findings": [],
    })

    AgentExceptionTriage(config_path=_config()).process(run_dir)
    approval = _load(run_dir / "approval_packet.json")
    categories = [e["category"] for e in approval["exceptions"]]
    assert "compliance" in categories
    assert "anomaly_review" in categories
