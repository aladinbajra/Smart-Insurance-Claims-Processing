from pathlib import Path

from backend.agents.agent_fnol_intake import AgentFNOLIntake
from backend.agents.agent_fraud_detection import AgentFraudDetection


def test_agent_e_processes_clean_auto_claim(tmp_path: Path) -> None:
    """
    Confirm that Agent E produces a low-risk fraud score
    for the shared clean auto-claim dataset.

    The test first runs Agent A to create context_packet.json,
    then runs Agent E against that generated run folder.
    """

    project_root = Path(__file__).resolve().parents[1]

    dataset_dir = (
        project_root
        / "datasets"
        / "SICPS_Dataset (3)"
        / "SICPS_Dataset"
        / "scenario_01_clean_auto"
    )

    runs_dir = tmp_path / "runs"

    agent_a = AgentFNOLIntake(
        config_path=project_root / "config" / "policy.yaml",
        runs_dir=runs_dir,
    )

    context_result = agent_a.process(dataset_dir)

    run_dir = runs_dir / context_result["claim_id"]

    agent_e = AgentFraudDetection(
        config_path=project_root / "config" / "policy.yaml"
    )

    fraud_result = agent_e.process(run_dir)

    assert fraud_result["claim_id"] == "CLM-2026-001"
    assert fraud_result["fraud_score"] == 0.0
    assert fraud_result["risk_level"] == "low"
    assert fraud_result["siu_referral"] is False
    assert fraud_result["adjuster_review"] is False
    assert fraud_result["provider_blacklisted"] is False
    assert fraud_result["provider_ring_match"] is False
    assert fraud_result["prior_claims_trigger"] is False
    assert fraud_result["indicators"] == []

    assert (run_dir / "context_packet.json").exists()
    assert (run_dir / "fraud_score.json").exists()
