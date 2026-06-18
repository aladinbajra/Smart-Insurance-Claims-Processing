from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dataset_root() -> Path:
    return _root() / "datasets" / "SICPS_Dataset (3)" / "SICPS_Dataset"


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    app = create_app(
        config_path=_root() / "config" / "policy.yaml",
        runs_dir=tmp_path / "runs",
        claims_dir=_dataset_root(),
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# Basic endpoints
# ---------------------------------------------------------------------------

def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_claims(client: TestClient) -> None:
    resp = client.get("/claims")
    assert resp.status_code == 200
    claims = resp.json()
    assert len(claims) == 12
    ids = {c["claim_id"] for c in claims}
    assert "CLM-2026-001" in ids
    assert {"claim_id", "scenario", "claim_type", "bundle"} <= claims[0].keys()


def test_policy_endpoint(client: TestClient) -> None:
    resp = client.get("/policy")
    assert resp.status_code == 200
    policy = resp.json()
    assert policy["approval"]["auto_settle_max_amount"] == 5000.0


# ---------------------------------------------------------------------------
# Run + retrieve
# ---------------------------------------------------------------------------

def test_run_and_fetch_result(client: TestClient) -> None:
    resp = client.post("/run", json={"bundle": "scenario_01_clean_auto"})
    assert resp.status_code == 200
    summary = resp.json()
    assert summary["claim_id"] == "CLM-2026-001"
    assert summary["routing_decision"] == "auto_settle"
    assert summary["net_settlement"] == 550.0
    assert summary["cache_hit"] is False

    # Listed as a completed run.
    assert "CLM-2026-001" in client.get("/runs").json()

    # Detail view exposes approval packet + metrics + artifact index.
    detail = client.get("/runs/CLM-2026-001").json()
    assert detail["routing_decision"] == "auto_settle"
    assert detail["metrics"]["routing_decision"] == "auto_settle"
    assert "approval_packet.json" in detail["artifacts"]
    assert "audit_log.md" in detail["artifacts"]


def test_run_fnol_single_document(client: TestClient) -> None:
    resp = client.post(
        "/run-fnol", json={"fnol": "scenario_01_clean_auto/fnol.pdf"}
    )
    assert resp.status_code == 200
    summary = resp.json()
    assert summary["claim_id"] == "CLM-2026-001"
    assert summary["routing_decision"] == "manual_review_route"
    assert summary["net_settlement"] == 0.0


def test_run_fnol_missing_404(client: TestClient) -> None:
    resp = client.post("/run-fnol", json={"fnol": "scenario_01_clean_auto/nope.pdf"})
    assert resp.status_code == 404


def test_second_run_is_cache_hit(client: TestClient) -> None:
    client.post("/run", json={"bundle": "scenario_07_fraud_ring"})
    resp = client.post("/run", json={"bundle": "scenario_07_fraud_ring"})
    body = resp.json()
    assert body["routing_decision"] == "siu_referral"
    assert body["cache_hit"] is True


def test_get_artifacts(client: TestClient) -> None:
    client.post("/run", json={"bundle": "scenario_07_fraud_ring"})

    md = client.get("/runs/CLM-2026-007/artifacts/exceptions.md").json()
    assert md["kind"] == "markdown"
    assert "siu_referral" in md["content"]

    js = client.get("/runs/CLM-2026-007/artifacts/approval_packet.json").json()
    assert js["kind"] == "json"
    assert js["content"]["routing_decision"] == "siu_referral"


# ---------------------------------------------------------------------------
# Error handling + security
# ---------------------------------------------------------------------------

def test_unknown_bundle_404(client: TestClient) -> None:
    resp = client.post("/run", json={"bundle": "does_not_exist"})
    assert resp.status_code == 404


def test_path_traversal_blocked(client: TestClient) -> None:
    resp = client.post("/run", json={"bundle": "../../../etc"})
    assert resp.status_code == 400


def test_unknown_run_404(client: TestClient) -> None:
    assert client.get("/runs/CLM-9999-999").status_code == 404


def test_unknown_artifact_400(client: TestClient) -> None:
    client.post("/run", json={"bundle": "scenario_01_clean_auto"})
    resp = client.get("/runs/CLM-2026-001/artifacts/secrets.env")
    assert resp.status_code == 400
