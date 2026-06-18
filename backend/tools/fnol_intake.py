"""
Single-FNOL intake — Spec §1a / Deliverable: accept a lone FNOL document
(not a full Claims Bundle) and synthesize the minimal claims bundle the
deterministic pipeline expects.

A standalone FNOL carries claimant / incident / policy-number context but none
of the supporting documents (policy, repair estimate, vendor master). The
synthesized manifest therefore declares those required documents as MISSING, so
the pipeline correctly routes the claim to manual review rather than fabricating
a settlement. Output is deterministic (no timestamps / randomness) so re-running
the same FNOL is idempotent.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from backend.tools.pdf_extractor import extract_document, load_policy


def _field_map(fnol_path: Path, config_path: Path) -> dict[str, Any]:
    """Extract FNOL fields into a flat name -> value map (best effort)."""
    policy = load_policy(config_path)
    fields = extract_document(fnol_path, "fnol", policy)
    out: dict[str, Any] = {}
    for ef in fields:
        if ef.value not in (None, "", []):
            out[ef.field_name] = ef.value
    return out


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"yes", "true", "y", "1"}


def synthesize_fnol_bundle(
    fnol_path: str | Path,
    dest_dir: str | Path,
    config_path: str | Path,
) -> Path:
    """
    Copy ``fnol_path`` into ``dest_dir`` as ``fnol.pdf`` and write a minimal
    ``manifest.yaml`` derived from the FNOL. Returns ``dest_dir`` (a valid
    single-document claims bundle).
    """
    fnol = Path(fnol_path)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if not fnol.is_file():
        raise FileNotFoundError(f"FNOL document not found: {fnol}")

    shutil.copyfile(fnol, dest / "fnol.pdf")

    fm = _field_map(fnol, Path(config_path))

    claim_id = str(fm.get("claim_id") or f"FNOL-{fnol.stem}")
    claim_type = str(fm.get("claim_type") or "auto").lower()
    incident_date = str(fm.get("incident_date") or "")
    report_date = str(fm.get("report_date") or incident_date)

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "claim_id": claim_id,
        "scenario": "single_fnol_intake",
        "claim_type": claim_type,
        "priority": "normal",
        "expected_outcome": "manual_review",
        "expected_routing": "manual_review_route",
        "claimant": {
            "claimant_id": str(fm.get("claimant_id") or "UNKNOWN"),
            "full_name": str(fm.get("claimant_name") or "UNKNOWN"),
            "policy_number": str(fm.get("policy_number") or "UNKNOWN"),
            "prior_claims_18mo": int(float(fm.get("prior_claims_18mo") or 0)),
            "prior_claim_ids": [],
            "fraud_history": _to_bool(fm.get("fraud_history")),
            "new_claimant": _to_bool(fm.get("new_claimant")),
        },
        # Policy details are unknown from an FNOL alone — use permissive
        # defaults so coverage validation does not falsely deny, while the
        # missing policy document forces a manual review downstream.
        "policy": {
            "policy_number": str(fm.get("policy_number") or "UNKNOWN"),
            "policy_type": claim_type,
            "effective_date": "1970-01-01",
            "expiration_date": "2999-12-31",
            "is_active": True,
            "policy_expired": False,
            "deductible": 0.0,
            "coverage_limit": 0.0,
            "late_notice_days": 36500,
            "exclusions": [],
            "state": "US",
        },
        "incident": {
            "date": incident_date or "1970-01-01",
            "report_date": report_date or "1970-01-01",
            "days_to_report": 0,
            "late_notice_violation": False,
            "location": str(fm.get("incident_location") or ""),
            "zip_code": str(fm.get("zip_code") or ""),
            "type": str(fm.get("incident_type") or "unknown"),
            "severity": "unknown",
            "cat_event": _to_bool(fm.get("cat_event")),
            "police_report_filed": _to_bool(fm.get("police_report")),
            "witness_count": int(float(fm.get("witness_count") or 0)),
        },
        # No financial documents accompany a lone FNOL.
        "financials": {
            "repair_estimate": 0.0,
            "medical_total": 0.0,
            "gross_amount": 0.0,
            "deductible": 0.0,
            "net_settlement": 0.0,
            "market_value": 0.0,
            "total_loss_ratio": 0.0,
            "total_loss_flag": False,
            "injury_claim": 0.0,
        },
        "documents": [
            {
                "type": "fnol",
                "file": "fnol.pdf",
                "processable": True,
                "expected_confidence": "high",
                "required": True,
            },
            {
                "type": "policy",
                "file": "policy.pdf",
                "processable": False,
                "expected_confidence": "not_applicable",
                "required": True,
                "status": "MISSING",
            },
            {
                "type": "repair_estimate",
                "file": "repair_estimate.pdf",
                "processable": False,
                "expected_confidence": "not_applicable",
                "required": True,
                "status": "MISSING",
            },
        ],
        "flags": {
            "cat_event": _to_bool(fm.get("cat_event")),
            "fraud_history": _to_bool(fm.get("fraud_history")),
            "new_claimant": _to_bool(fm.get("new_claimant")),
            "late_notice": False,
            "coverage_denied": False,
            "siu_referral": False,
            "manual_review": True,
            "total_loss": False,
            "missing_documents": True,
            "duplicate_claim": False,
            "policy_expired": False,
            "low_ocr_confidence": False,
        },
        "adjuster_notes": (
            "Single-FNOL intake: supporting bundle documents (policy, repair "
            "estimate) are not yet submitted."
        ),
    }

    with (dest / "manifest.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=True, default_flow_style=False)

    return dest
