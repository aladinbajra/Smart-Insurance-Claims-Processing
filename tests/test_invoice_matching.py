from __future__ import annotations

import json
from pathlib import Path

from backend.pipeline.runner import PipelineRunner
from backend.tools.invoice_matcher import match_invoice


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dataset(scenario: str) -> Path:
    return _root() / "datasets" / "SICPS_Dataset (3)" / "SICPS_Dataset" / scenario


def _runner(tmp_path: Path) -> PipelineRunner:
    return PipelineRunner(
        config_path=_root() / "config" / "policy.yaml",
        runs_dir=tmp_path / "runs",
    )


# ---------------------------------------------------------------------------
# Unit tests — the matcher derives status, never reads a flag (REQ-043)
# ---------------------------------------------------------------------------

def test_match_within_tolerance() -> None:
    r = match_invoice(1550.0, 1550.0, 0.04, "PO-001", None, tolerance=0.25)
    assert r.match_status == "match"
    assert r.billing_variance_pct == 0.04
    assert r.po_present is True


def test_mismatch_on_high_variance() -> None:
    r = match_invoice(45900.0, 45900.0, 1.87, "PO-007", None, tolerance=0.25)
    assert r.match_status == "mismatch"
    assert r.billing_variance_pct == 1.87
    assert any("variance" in reason for reason in r.reasons)


def test_mismatch_on_missing_po() -> None:
    r = match_invoice(1000.0, 1000.0, 0.0, None, None, tolerance=0.25)
    assert r.match_status == "mismatch"
    assert "purchase-order" in " ".join(r.reasons)


def test_total_variance_detected_even_without_line_variance() -> None:
    # Billed total far above the PO/expected total -> mismatch.
    r = match_invoice(2000.0, 1000.0, 0.0, "PO-X", None, tolerance=0.25)
    assert r.match_status == "mismatch"
    assert r.billing_variance_pct == 1.0


# ---------------------------------------------------------------------------
# Pipeline tests — matching surfaces in settlement_calc + approval
# ---------------------------------------------------------------------------

def test_clean_invoice_matches(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run(_dataset("scenario_01_clean_auto"))
    settlement = json.loads((result.run_dir / "settlement_calc.json").read_text())

    assert settlement["invoice_match_status"] == "match"
    assert settlement["billing_variance_pct"] == 0.04


def test_fraud_ring_invoice_mismatch(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run(_dataset("scenario_07_fraud_ring"))
    settlement = json.loads((result.run_dir / "settlement_calc.json").read_text())
    approval = json.loads((result.run_dir / "approval_packet.json").read_text())

    # Inflated billing (1.87) is detected as a mismatch (REQ-043).
    assert settlement["invoice_match_status"] == "mismatch"
    assert settlement["billing_variance_pct"] == 1.87

    # The mismatch is recorded as a billing exception alongside the fraud one.
    categories = [e["category"] for e in approval["exceptions"]]
    assert "billing_variance" in categories
    # A D-008 invoice-mismatch finding is in the merged findings.
    assert any(
        f["finding_id"].startswith("D-008") for f in approval["all_findings"]
    )
