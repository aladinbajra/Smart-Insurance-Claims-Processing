"""
SICPS Backend API — FastAPI service (REQ-054).

Thin HTTP layer over the deterministic pipeline runner. It exposes the claim
bundles, triggers pipeline runs, and serves the resulting run artifacts so the
Streamlit frontend (or any client) can drive the demo:

    GET  /health                              liveness
    GET  /claims                              discoverable claim bundles
    POST /run                                 run the pipeline for one bundle
    GET  /runs                                completed run ids
    GET  /runs/{claim_id}                     approval packet + metrics + index
    GET  /runs/{claim_id}/artifacts/{name}    a single run artifact
    GET  /policy                              policy pack (configurability demo)

Configuration is environment-driven (REQ-035) with project-relative defaults so
`uvicorn backend.main:app` works out of the box.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.pipeline.runner import PipelineRunner

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "policy.yaml"
_DEFAULT_RUNS = _PROJECT_ROOT / "runs"
_DEFAULT_CLAIMS = (
    _PROJECT_ROOT / "datasets" / "SICPS_Dataset (3)" / "SICPS_Dataset"
)

# Artifacts the API is willing to serve (whitelist — no arbitrary file reads).
_JSON_ARTIFACTS = {
    "context_packet.json",
    "claim_summary.json",
    "coverage_result.json",
    "settlement_calc.json",
    "fraud_score.json",
    "approval_packet.json",
    "metrics.json",
    "evidence_index.json",
}
_TEXT_ARTIFACTS = {"exceptions.md", "audit_log.md"}
_ARTIFACTS = _JSON_ARTIFACTS | _TEXT_ARTIFACTS


# ----------------------------------------------------------------------
# Response models
# ----------------------------------------------------------------------

class ClaimBundle(BaseModel):
    claim_id: str
    scenario: str
    claim_type: str
    priority: str
    expected_outcome: str
    bundle: str


class RunSummary(BaseModel):
    claim_id: str
    routing_decision: str
    net_settlement: float
    cache_hit: bool
    duration_seconds: float
    input_hash: str


class RunDetail(BaseModel):
    claim_id: str
    routing_decision: str
    net_settlement: float
    approval_packet: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: list[str]


class ArtifactResponse(BaseModel):
    name: str
    kind: str  # "json" | "markdown"
    content: Any


class RunRequest(BaseModel):
    bundle: str
    force: bool = False


class FnolRequest(BaseModel):
    fnol: str  # path to a single FNOL document, relative to the claims dir
    force: bool = False


# ----------------------------------------------------------------------
# App factory
# ----------------------------------------------------------------------

def create_app(
    config_path: str | Path | None = None,
    runs_dir: str | Path | None = None,
    claims_dir: str | Path | None = None,
) -> FastAPI:
    config_path = Path(config_path or os.getenv("POLICY_PACK_PATH", _DEFAULT_CONFIG))
    runs_dir = Path(runs_dir or os.getenv("RUNS_DIR", _DEFAULT_RUNS))
    claims_dir = Path(claims_dir or os.getenv("SICPS_CLAIMS_DIR", _DEFAULT_CLAIMS))

    runner = PipelineRunner(config_path=config_path, runs_dir=runs_dir)

    app = FastAPI(title="SICPS API", version="1.0", description="Smart Insurance Claims Processing System")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------- helpers -------------

    def _resolve_bundle(name: str) -> Path:
        candidate = (claims_dir / name).resolve()
        claims_root = claims_dir.resolve()
        if claims_root != candidate and claims_root not in candidate.parents:
            raise HTTPException(status_code=400, detail="invalid bundle path")
        if not (candidate / "manifest.yaml").is_file():
            raise HTTPException(status_code=404, detail=f"bundle not found: {name}")
        return candidate

    def _resolve_fnol(name: str) -> Path:
        candidate = (claims_dir / name).resolve()
        claims_root = claims_dir.resolve()
        if claims_root not in candidate.parents:
            raise HTTPException(status_code=400, detail="invalid fnol path")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"fnol not found: {name}")
        return candidate

    def _run_dir(claim_id: str) -> Path:
        run_dir = (runs_dir / claim_id).resolve()
        if runs_dir.resolve() not in run_dir.parents:
            raise HTTPException(status_code=400, detail="invalid claim id")
        return run_dir

    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    # ------------- routes -------------

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": "SICPS API",
            "version": "1.0",
            "endpoints": [
                "/health",
                "/claims",
                "/run",
                "/run-fnol",
                "/runs",
                "/runs/{claim_id}",
                "/runs/{claim_id}/artifacts/{name}",
                "/policy",
            ],
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/claims", response_model=list[ClaimBundle])
    def list_claims() -> list[ClaimBundle]:
        bundles: list[ClaimBundle] = []
        for manifest_path in sorted(claims_dir.glob("*/manifest.yaml")):
            try:
                with manifest_path.open("r", encoding="utf-8") as file:
                    manifest = yaml.safe_load(file) or {}
            except (OSError, yaml.YAMLError):
                continue
            bundles.append(
                ClaimBundle(
                    claim_id=str(manifest.get("claim_id", "")),
                    scenario=str(manifest.get("scenario", manifest_path.parent.name)),
                    claim_type=str(manifest.get("claim_type", "")),
                    priority=str(manifest.get("priority", "")),
                    expected_outcome=str(manifest.get("expected_outcome", "")),
                    bundle=manifest_path.parent.name,
                )
            )
        return bundles

    @app.post("/run", response_model=RunSummary)
    def run_pipeline(request: RunRequest) -> RunSummary:
        bundle_path = _resolve_bundle(request.bundle)
        try:
            result = runner.run(bundle_path, force=request.force)
        except Exception as exc:  # noqa: BLE001 — surface as HTTP error
            raise HTTPException(
                status_code=500, detail=f"pipeline failed: {exc}"
            ) from exc
        return RunSummary(
            claim_id=result.claim_id,
            routing_decision=result.routing_decision,
            net_settlement=result.net_settlement,
            cache_hit=result.cache_hit,
            duration_seconds=result.duration_seconds,
            input_hash=result.input_hash,
        )

    @app.post("/run-fnol", response_model=RunSummary)
    def run_fnol(request: FnolRequest) -> RunSummary:
        """Single-FNOL intake (Spec §1a) — run the pipeline from one FNOL doc."""
        fnol_path = _resolve_fnol(request.fnol)
        try:
            result = runner.run_fnol(fnol_path, force=request.force)
        except Exception as exc:  # noqa: BLE001 — surface as HTTP error
            raise HTTPException(
                status_code=500, detail=f"fnol intake failed: {exc}"
            ) from exc
        return RunSummary(
            claim_id=result.claim_id,
            routing_decision=result.routing_decision,
            net_settlement=result.net_settlement,
            cache_hit=result.cache_hit,
            duration_seconds=result.duration_seconds,
            input_hash=result.input_hash,
        )

    @app.get("/runs", response_model=list[str])
    def list_runs() -> list[str]:
        if not runs_dir.is_dir():
            return []
        return sorted(
            p.name
            for p in runs_dir.iterdir()
            if (p / "approval_packet.json").is_file()
        )

    @app.get("/runs/{claim_id}", response_model=RunDetail)
    def get_run(claim_id: str) -> RunDetail:
        run_dir = _run_dir(claim_id)
        approval_path = run_dir / "approval_packet.json"
        if not approval_path.is_file():
            raise HTTPException(status_code=404, detail=f"no run for {claim_id}")

        approval = _load_json(approval_path)
        metrics_path = run_dir / "metrics.json"
        metrics = _load_json(metrics_path) if metrics_path.is_file() else {}
        artifacts = sorted(n for n in _ARTIFACTS if (run_dir / n).is_file())

        return RunDetail(
            claim_id=claim_id,
            routing_decision=str(approval.get("routing_decision", "")),
            net_settlement=float(approval.get("net_settlement", 0.0)),
            approval_packet=approval,
            metrics=metrics,
            artifacts=artifacts,
        )

    @app.get("/runs/{claim_id}/artifacts/{name}", response_model=ArtifactResponse)
    def get_artifact(claim_id: str, name: str) -> ArtifactResponse:
        if name not in _ARTIFACTS:
            raise HTTPException(status_code=400, detail=f"unknown artifact: {name}")
        path = _run_dir(claim_id) / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"artifact not found: {name}")
        if name in _JSON_ARTIFACTS:
            return ArtifactResponse(name=name, kind="json", content=_load_json(path))
        return ArtifactResponse(
            name=name, kind="markdown", content=path.read_text(encoding="utf-8")
        )

    @app.get("/policy")
    def get_policy() -> dict[str, Any]:
        if not config_path.is_file():
            raise HTTPException(status_code=404, detail="policy pack not found")
        with config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    return app


# Module-level app for `uvicorn backend.main:app`.
app = create_app()
