from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from backend.pipeline.runner import PipelineRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dataset(scenario: str) -> Path:
    return _root() / "datasets" / "SICPS_Dataset (3)" / "SICPS_Dataset" / scenario


def _config() -> Path:
    return _root() / "config" / "policy.yaml"


def _runner(tmp_path: Path) -> PipelineRunner:
    return PipelineRunner(config_path=_config(), runs_dir=tmp_path / "runs")


_ARTIFACTS = (
    "context_packet.json",
    "claim_summary.json",
    "coverage_result.json",
    "settlement_calc.json",
    "fraud_score.json",
    "approval_packet.json",
    "exceptions.md",
    "audit_log.md",
    "metrics.json",
)


# ---------------------------------------------------------------------------
# Full pipeline produces every artifact and the right decision (REQ-007/029)
# ---------------------------------------------------------------------------

def test_run_produces_all_artifacts(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run(_dataset("scenario_01_clean_auto"))

    assert result.claim_id == "CLM-2026-001"
    assert result.routing_decision == "auto_settle"
    assert result.net_settlement == 550.0
    assert result.cache_hit is False
    assert result.duration_seconds > 0.0

    for artifact in _ARTIFACTS:
        assert (result.run_dir / artifact).exists(), artifact


# ---------------------------------------------------------------------------
# Idempotency cache — second run is served from cache (REQ-028)
# ---------------------------------------------------------------------------

def test_second_run_is_cache_hit(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    first = runner.run(_dataset("scenario_07_fraud_ring"))
    approval_bytes = (first.run_dir / "approval_packet.json").read_bytes()

    second = runner.run(_dataset("scenario_07_fraud_ring"))

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.routing_decision == first.routing_decision
    # Cached run does not rewrite artifacts.
    assert (second.run_dir / "approval_packet.json").read_bytes() == approval_bytes


# ---------------------------------------------------------------------------
# REQ-044 — forced re-run is byte-for-byte identical across all artifacts
# ---------------------------------------------------------------------------

def test_determinism_byte_for_byte(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    assert runner.verify_determinism(_dataset("scenario_01_clean_auto")) is True
    assert runner.verify_determinism(_dataset("scenario_07_fraud_ring")) is True


# ---------------------------------------------------------------------------
# Batch processing of multiple bundles (REQ-040)
# ---------------------------------------------------------------------------

def test_run_all_batch(tmp_path: Path) -> None:
    claims_dir = tmp_path / "claims"
    claims_dir.mkdir()
    for scenario in ("scenario_01_clean_auto", "scenario_07_fraud_ring"):
        shutil.copytree(_dataset(scenario), claims_dir / scenario)

    runner = _runner(tmp_path)
    results = runner.run_all(claims_dir)

    assert len(results) == 2
    by_id = {r.claim_id: r for r in results}
    assert by_id["CLM-2026-001"].routing_decision == "auto_settle"
    assert by_id["CLM-2026-007"].routing_decision == "siu_referral"
    assert all(r.error is None for r in results)


# ---------------------------------------------------------------------------
# Cache invalidation — changing the bundle forces a recompute (REQ-044)
# ---------------------------------------------------------------------------

def test_cache_invalidated_on_input_change(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(_dataset("scenario_01_clean_auto"), bundle)

    runner = _runner(tmp_path)
    assert runner.run(bundle).cache_hit is False
    assert runner.run(bundle).cache_hit is True  # unchanged -> cache

    # Mutate the input bundle (a YAML comment changes bytes, not parsed data).
    manifest = bundle / "manifest.yaml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "\n# touched\n", encoding="utf-8"
    )

    assert runner.run(bundle).cache_hit is False  # hash changed -> recompute


# ---------------------------------------------------------------------------
# Missing manifest raises a clear error
# ---------------------------------------------------------------------------

def test_missing_manifest_errors(tmp_path: Path) -> None:
    empty = tmp_path / "empty_bundle"
    empty.mkdir()
    runner = _runner(tmp_path)
    try:
        runner.run(empty)
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# REQ-044 — every dataset scenario is byte-for-byte deterministic end-to-end
# ---------------------------------------------------------------------------

_ALL_SCENARIOS = [
    "scenario_01_clean_auto",
    "scenario_02_total_loss",
    "scenario_03_siu_referral",
    "scenario_04_coverage_denial",
    "scenario_05_late_notice",
    "scenario_06_cat_event",
    "scenario_07_fraud_ring",
    "scenario_08_low_ocr",
    "scenario_09_homeowners",
    "scenario_10_expired_policy",
    "scenario_11_missing_documents",
    "scenario_12_duplicate_claim",
]


@pytest.mark.parametrize("scenario", _ALL_SCENARIOS)
def test_every_scenario_is_deterministic(tmp_path: Path, scenario: str) -> None:
    runner = _runner(tmp_path)
    assert runner.verify_determinism(_dataset(scenario)) is True

    # All eight REQ-029 artifacts must be present after a run.
    result = runner.run(_dataset(scenario), force=True)
    for artifact in _ARTIFACTS:
        assert (result.run_dir / artifact).exists(), f"{scenario}:{artifact}"
