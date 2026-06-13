from pathlib import Path

from backend.agents.agent_fnol_intake import AgentFNOLIntake


def test_agent_a_processes_clean_auto_claim(tmp_path: Path) -> None:
    """
    Confirm that Agent A processes the shared clean auto-claim dataset.

    The test uses a temporary output folder so generated artifacts
    are not written permanently inside the project's runs folder.
    """

    # Locate the main SICPS project folder.
    project_root = Path(__file__).resolve().parents[1]

    # Use the clean auto-claim bundle provided in the shared dataset.
    dataset_dir = (
        project_root
        / "datasets"
        / "SICPS_Dataset (3)"
        / "SICPS_Dataset"
        / "scenario_01_clean_auto"
    )

    # Store test artifacts in a temporary folder.
    runs_dir = tmp_path / "runs"

    # Create Agent A using the shared policy configuration.
    agent = AgentFNOLIntake(
        config_path=project_root / "config" / "policy.yaml",
        runs_dir=runs_dir,
    )

    # Process the clean auto-claim bundle.
    result = agent.process(dataset_dir)

    # Confirm the basic claim information.
    assert result["claim_id"] == "CLM-2026-001"
    assert result["claim_type"] == "auto"

    # A clean claim should continue to Agent B without intake warnings.
    assert result["status"] == "ready_for_document_extraction"
    assert result["next_agent"] == "agent_b"
    assert result["findings"] == []

    # Confirm that all seven evidence references were indexed.
    assert len(result["evidence_index"]) == 7

    # Confirm that the reference-only photos remain indexed correctly.
    photo_entries = [
        item
        for item in result["evidence_index"]
        if item["doc_type"] == "photo"
    ]

    assert len(photo_entries) == 2
    assert all(
        item["processable"] is False
        for item in photo_entries
    )

    # Confirm that Agent A generated its required run artifacts.
    output_dir = runs_dir / "CLM-2026-001"

    assert (output_dir / "context_packet.json").exists()
    assert (output_dir / "evidence_index.json").exists()
    assert (output_dir / "audit_log.md").exists()
