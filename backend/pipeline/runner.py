"""
SICPS Pipeline Runner — REQ-007, REQ-038-041, REQ-044, REQ-054.

Orchestrates the deterministic 6-agent pipeline (A -> B -> C -> D -> E -> H)
over a claims bundle, using the shared run directory as the only channel
between agents (file-based coordination, REQ-038/039).

Idempotency (REQ-028/044): the runner computes a SHA-256 over the input bundle
and caches it next to the run artifacts. Re-running an unchanged bundle is a
cache hit and skips recomputation; the produced artifacts are byte-for-byte
reproducible on a forced re-run.

Determinism note: wall-clock pipeline duration is inherently non-reproducible,
so it is reported at runtime (RunResult / CLI) rather than written into
metrics.json. Every persisted artifact therefore stays byte-for-byte stable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from backend.agents.agent_coverage_validation import AgentCoverageValidation
from backend.agents.agent_damage_assessment import AgentDamageAssessment
from backend.agents.agent_document_extraction import AgentDocumentExtraction
from backend.agents.agent_exception_triage import AgentExceptionTriage
from backend.agents.agent_fnol_intake import AgentFNOLIntake
from backend.agents.agent_fraud_detection import AgentFraudDetection


_CACHE_MARKER = ".pipeline_input_hash"

# Artifacts whose byte-for-byte stability the determinism check verifies
# (REQ-044). The cache marker and per-agent scratch logs are excluded.
_DETERMINISTIC_ARTIFACTS = (
    "context_packet.json",
    "claim_summary.json",
    "coverage_result.json",
    "settlement_calc.json",
    "fraud_score.json",
    "approval_packet.json",
    "exceptions.md",
    "audit_log.md",
    "metrics.json",
    "settlement_payload.csv",
)


@dataclass
class RunResult:
    """Outcome of a single pipeline run."""

    claim_id: str
    routing_decision: str
    net_settlement: float
    cache_hit: bool
    duration_seconds: float
    input_hash: str
    run_dir: Path
    error: str | None = None


class PipelineRunner:
    """
    Deterministic orchestrator for the SICPS 6-agent pipeline.

    Agents communicate only through files in ``runs/<claim_id>/`` — the runner
    sequences them and adapts their differing call signatures, but never passes
    in-memory state between agents (REQ-038/040).
    """

    def __init__(
        self,
        config_path: str | Path,
        runs_dir: str | Path,
    ) -> None:
        self.config_path = Path(config_path)
        self.runs_dir = Path(runs_dir)

        # Agents are stateless across claims; instantiate once and reuse.
        self.agent_a = AgentFNOLIntake(self.config_path, self.runs_dir)
        self.agent_b = AgentDocumentExtraction(self.config_path, self.runs_dir)
        self.agent_c = AgentCoverageValidation(self.runs_dir, self.config_path)
        self.agent_d = AgentDamageAssessment(self.config_path, self.runs_dir)
        self.agent_e = AgentFraudDetection(self.config_path)
        self.agent_h = AgentExceptionTriage(self.config_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, bundle_dir: str | Path, force: bool = False) -> RunResult:
        """
        Run the full pipeline for one claims bundle.

        With ``force=False`` an unchanged bundle (same SHA-256) is served from
        cache without recomputation (REQ-028). With ``force=True`` every agent
        re-runs; the artifacts are byte-for-byte identical (REQ-044).
        """
        bundle = Path(bundle_dir)
        manifest_path = bundle / "manifest.yaml"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest.yaml not found: {manifest_path}")

        claim_id = str(self._load_yaml(manifest_path).get("claim_id", ""))
        if not claim_id:
            raise ValueError(f"manifest.yaml has no claim_id: {manifest_path}")

        run_dir = self.runs_dir / claim_id
        input_hash = self._bundle_hash(bundle)

        # ----- Idempotency cache (REQ-028) -----
        if not force and self._is_cached(run_dir, input_hash):
            approval = self._load_json(run_dir / "approval_packet.json")
            return RunResult(
                claim_id=claim_id,
                routing_decision=str(approval.get("routing_decision", "")),
                net_settlement=float(approval.get("net_settlement", 0.0)),
                cache_hit=True,
                duration_seconds=0.0,
                input_hash=input_hash,
                run_dir=run_dir,
            )

        # ----- Execute the pipeline A -> B -> C -> D -> E -> H -----
        start = time.perf_counter()

        self.agent_a.process(bundle)
        self.agent_b.process(bundle)

        claim_summary = self._load_json(run_dir / "claim_summary.json")
        policy = self._load_json(run_dir / "context_packet.json").get("policy", {})
        self.agent_c.process(claim_summary, policy)

        coverage_result = self._load_json(run_dir / "coverage_result.json")
        self.agent_d.process(claim_summary, coverage_result)

        self.agent_e.process(run_dir)

        # duration is reported at runtime, not persisted, to keep metrics.json
        # byte-for-byte deterministic (REQ-044).
        approval = self.agent_h.process(run_dir, pipeline_duration_seconds=0.0)

        duration = time.perf_counter() - start

        self._write_cache_marker(run_dir, input_hash)

        return RunResult(
            claim_id=claim_id,
            routing_decision=str(approval.get("routing_decision", "")),
            net_settlement=float(approval.get("net_settlement", 0.0)),
            cache_hit=False,
            duration_seconds=round(duration, 4),
            input_hash=input_hash,
            run_dir=run_dir,
        )

    def run_all(self, claims_dir: str | Path, force: bool = False) -> list[RunResult]:
        """
        Run every claims bundle under ``claims_dir`` (each subfolder with a
        manifest.yaml). One bundle failing does not abort the batch.
        """
        claims_path = Path(claims_dir)
        results: list[RunResult] = []

        for manifest_path in sorted(claims_path.glob("*/manifest.yaml")):
            bundle = manifest_path.parent
            try:
                results.append(self.run(bundle, force=force))
            except Exception as exc:  # noqa: BLE001 — batch resilience (REQ-040)
                claim_id = str(self._load_yaml(manifest_path).get("claim_id", bundle.name))
                results.append(
                    RunResult(
                        claim_id=claim_id,
                        routing_decision="error",
                        net_settlement=0.0,
                        cache_hit=False,
                        duration_seconds=0.0,
                        input_hash="",
                        run_dir=self.runs_dir / claim_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return results

    def verify_determinism(self, bundle_dir: str | Path) -> bool:
        """
        Run the pipeline twice (forced) and confirm every persisted artifact is
        byte-for-byte identical — the REQ-044 success-criteria checkpoint.
        """
        first = self.run(bundle_dir, force=True)
        digest_first = self._artifacts_digest(first.run_dir)

        second = self.run(bundle_dir, force=True)
        digest_second = self._artifacts_digest(second.run_dir)

        return digest_first == digest_second

    # ------------------------------------------------------------------
    # Cache / hashing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bundle_hash(bundle: Path) -> str:
        """Deterministic SHA-256 over all input files (sorted path + content)."""
        digest = hashlib.sha256()
        for file_path in sorted(p for p in bundle.rglob("*") if p.is_file()):
            digest.update(str(file_path.relative_to(bundle)).encode("utf-8"))
            digest.update(file_path.read_bytes())
        return f"sha256:{digest.hexdigest()}"

    def _is_cached(self, run_dir: Path, input_hash: str) -> bool:
        marker = run_dir / _CACHE_MARKER
        approval = run_dir / "approval_packet.json"
        if not marker.is_file() or not approval.is_file():
            return False
        return marker.read_text(encoding="utf-8").strip() == input_hash

    @staticmethod
    def _write_cache_marker(run_dir: Path, input_hash: str) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / _CACHE_MARKER).write_text(input_hash + "\n", encoding="utf-8")

    @staticmethod
    def _artifacts_digest(run_dir: Path) -> str:
        digest = hashlib.sha256()
        for name in _DETERMINISTIC_ARTIFACTS:
            path = run_dir / name
            if path.is_file():
                digest.update(name.encode("utf-8"))
                digest.update(path.read_bytes())
        return digest.hexdigest()

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)


# ----------------------------------------------------------------------
# CLI — single-command demo entry point (REQ-054)
# ----------------------------------------------------------------------

def _format(result: RunResult) -> str:
    if result.error:
        return f"  [ERROR] {result.claim_id}: {result.error}"
    tag = "cache" if result.cache_hit else f"{result.duration_seconds:.3f}s"
    return (
        f"  {result.claim_id}: {result.routing_decision} "
        f"(net ${result.net_settlement:,.2f}) [{tag}]"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the SICPS claims pipeline (Agents A-H)."
    )
    parser.add_argument(
        "path",
        help="Path to a claims bundle, or a directory of bundles with --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Treat 'path' as a directory containing multiple claim bundles.",
    )
    parser.add_argument("--runs-dir", default="runs", help="Run output directory.")
    parser.add_argument(
        "--config", default="config/policy.yaml", help="Policy pack path."
    )
    parser.add_argument(
        "--force", action="store_true", help="Ignore the idempotency cache."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run twice and confirm byte-for-byte determinism (REQ-044).",
    )
    args = parser.parse_args(argv)

    runner = PipelineRunner(config_path=args.config, runs_dir=args.runs_dir)

    if args.verify:
        ok = runner.verify_determinism(args.path)
        print(f"Determinism check: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    if args.all:
        results = runner.run_all(args.path, force=args.force)
        print(f"Processed {len(results)} claim(s):")
        for result in results:
            print(_format(result))
        return 0 if all(r.error is None for r in results) else 1

    result = runner.run(args.path, force=args.force)
    print("Pipeline complete:")
    print(_format(result))
    print(f"  artifacts: {result.run_dir}")
    return 0 if result.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
