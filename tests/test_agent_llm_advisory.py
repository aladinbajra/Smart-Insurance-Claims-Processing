from __future__ import annotations

import json
from pathlib import Path

from backend.agents.agent_llm_advisory import AgentLLMAdvisory


def _write_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    context = {
        "claim_id": "CLM-TEST-001",
        "claim_type": "auto",
        "incident": {"type": "collision", "severity": "moderate"},
        "policy": {"state": "IL"},
    }
    with (run_dir / "context_packet.json").open("w", encoding="utf-8") as fh:
        json.dump(context, fh)


# ---------------------------------------------------------------------------
# Graceful degradation — no API key must NOT crash; advisory is "unavailable"
# and the artifacts are still written. Keeps CI free of live API calls.
# ---------------------------------------------------------------------------

def test_advisory_degrades_without_api_key(tmp_path: Path, monkeypatch) -> None:
    # Another module's import-time load_dotenv() may have populated the env,
    # so clear the key to exercise the no-key degradation path deterministically.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    run_dir = tmp_path / "runs" / "CLM-TEST-001"
    _write_run(run_dir)

    result = AgentLLMAdvisory(api_key=None).process(run_dir)

    assert result["status"] == "unavailable"
    assert result["advisory_only"] is True
    assert result["claim_id"] == "CLM-TEST-001"
    assert (run_dir / "llm_advisory.json").exists()
    assert (run_dir / "llm_advisory.md").exists()
