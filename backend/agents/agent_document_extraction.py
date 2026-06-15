from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from backend.models.schemas import (
    ClaimSummary,
    ExtractedField,
    Finding,
    Severity,
)

# Reuse the shared PDF extractor created by the group.
# Agent B is the primary consumer of this tool — it turns raw PDFs
# into confidence-scored, bounding-box-traceable fields.
from backend.tools.pdf_extractor import extract_bundle, load_policy


class AgentDocumentExtraction:
    """
    Agent B: Document Extraction and Confidence Scoring.

    Second agent in the SICPS pipeline.

    Responsibilities:
    - Read the context packet produced by Agent A.
    - Extract every claim field from the submitted PDFs using the shared
      extractor (per-field confidence + bounding box for full traceability).
    - Aggregate fields across all documents (highest confidence wins).
    - Flag low-confidence extractions and missing required documents.
    - Denormalize the key fields downstream agents depend on.
    - Write a schema-validated claim_summary.json plus an audit trail.

    Agent B never halts the pipeline. It records review signals and always
    forwards the claim to Agent C. Final routing remains Agent H's job.
    """

    def __init__(
        self,
        config_path: str | Path,
        runs_dir: str | Path,
    ) -> None:
        # Shared policy configuration path.
        self.config_path = Path(config_path)

        # Folder where every agent reads and writes run artifacts.
        self.runs_dir = Path(runs_dir)

        # Plain YAML for simple threshold access.
        self.policy = self._load_yaml(self.config_path)

        # Validated policy object required by the shared PDF extractor.
        self.policy_pack = load_policy(self.config_path)

    def process(self, bundle_dir: str | Path) -> dict[str, Any]:
        """
        Process one SICPS claim bundle.

        Agent B creates:
        - runs/<claim_id>/claim_summary.json
        - runs/<claim_id>/audit_log_agent_b.md
        """

        bundle_path = Path(bundle_dir)
        manifest_path = bundle_path / "manifest.yaml"

        # A valid claims bundle must contain a manifest file.
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest file not found: {manifest_path}"
            )

        raw_manifest = self._load_yaml(manifest_path)
        claim_id = str(raw_manifest.get("claim_id", ""))

        run_dir = self.runs_dir / claim_id

        # Enforce pipeline order: Agent A must have run first.
        context_path = run_dir / "context_packet.json"
        if not context_path.is_file():
            raise FileNotFoundError(
                "context_packet.json not found. Agent A must run before Agent B: "
                f"{context_path}"
            )

        context = self._load_json(context_path)

        created_at = str(context.get("created_at", "1970-01-01T00:00:00Z"))

        audit_entries: list[str] = [
            "Agent B started document extraction.",
            "Context packet loaded from Agent A.",
        ]

        # ------------------------------------------------------------------
        # Core extraction — confidence + bbox per field, aggregated across docs.
        # ------------------------------------------------------------------
        extracted_map = extract_bundle(
            claims_dir=bundle_path,
            policy=self.policy_pack,
        )
        extracted_fields = list(extracted_map.values())

        audit_entries.append(
            f"Extracted {len(extracted_fields)} field(s) across the bundle."
        )

        # ------------------------------------------------------------------
        # Quality signals.
        # ------------------------------------------------------------------
        flag_threshold = self.policy_pack.extraction.low_confidence_flag_threshold
        min_auto = self.policy_pack.extraction.min_confidence_auto_process

        overall_confidence = (
            sum(ef.confidence for ef in extracted_fields) / len(extracted_fields)
            if extracted_fields
            else 0.0
        )

        low_confidence_fields = sorted(
            ef.field_name
            for ef in extracted_fields
            if ef.confidence < flag_threshold
        )

        # Document quality flag carried from the manifest (set by data source).
        low_ocr = bool(
            raw_manifest.get("flags", {}).get("low_ocr_confidence", False)
        )

        # Missing required documents are a hard extraction error.
        missing_required_docs = self._missing_required_documents(
            bundle_path=bundle_path,
            documents=raw_manifest.get("documents", []) or [],
        )

        extraction_errors: list[str] = [
            f"Required document missing or unreadable: {doc_type} ({filename})"
            for doc_type, filename in missing_required_docs
        ]

        # A claim needs manual review when overall confidence is too low,
        # the source documents are flagged low quality, or a required
        # document is missing. A single unmatched optional field
        # (e.g. an un-templated vendor form) must NOT trip this.
        requires_manual_review = (
            overall_confidence < min_auto
            or low_ocr
            or bool(missing_required_docs)
        )

        # ------------------------------------------------------------------
        # Denormalized fields — prefer the extracted value, fall back to the
        # Agent A context when extraction could not resolve it.
        # ------------------------------------------------------------------
        ctx_claimant = context.get("claimant", {}) or {}
        ctx_policy = context.get("policy", {}) or {}
        ctx_incident = context.get("incident", {}) or {}
        ctx_financials = context.get("financials", {}) or {}
        ctx_vehicle = context.get("vehicle") or {}
        ctx_provider = context.get("provider") or {}

        repair_estimate = self._resolve_float(
            extracted_map, "repair_estimate",
            ctx_financials.get("repair_estimate", 0.0),
        )
        medical_total = self._resolve_float(
            extracted_map, "medical_total",
            ctx_financials.get("medical_total", 0.0),
        )
        market_value = self._resolve_float(
            extracted_map, "market_value",
            ctx_vehicle.get("market_value", ctx_financials.get("market_value", 0.0)),
        )

        summary = ClaimSummary(
            claim_id=claim_id,
            extracted_fields=extracted_fields,
            overall_confidence=overall_confidence,
            requires_manual_review=requires_manual_review,
            low_confidence_fields=low_confidence_fields,
            extraction_errors=extraction_errors,
            findings=self._build_findings(
                claim_id=claim_id,
                overall_confidence=overall_confidence,
                min_auto=min_auto,
                low_ocr=low_ocr,
                low_confidence_fields=low_confidence_fields,
                missing_required_docs=missing_required_docs,
                created_at=created_at,
            ),
            # Denormalized for downstream agents.
            claimant_name=self._resolve_str(
                extracted_map, "claimant_name",
                ctx_claimant.get("full_name", ""),
            ),
            policy_number=self._resolve_str(
                extracted_map, "policy_number",
                ctx_policy.get("policy_number", ""),
            ),
            claim_type=str(context.get("claim_type", raw_manifest.get("claim_type", ""))),
            incident_date=self._resolve_str(
                extracted_map, "incident_date",
                ctx_incident.get("date", ""),
            ),
            repair_estimate=repair_estimate,
            medical_total=medical_total,
            gross_amount=round(repair_estimate + medical_total, 2),
            market_value=market_value,
            days_to_report=int(ctx_incident.get("days_to_report", 0)),
            zip_code=self._resolve_str(
                extracted_map, "zip_code",
                ctx_incident.get("zip_code", ""),
            ),
            prior_claims_18mo=int(ctx_claimant.get("prior_claims_18mo", 0)),
            vehicle_vin=self._resolve_optional_str(
                extracted_map, "vehicle_vin",
                ctx_vehicle.get("vin"),
            ),
            provider_id=self._resolve_optional_str(
                extracted_map, "provider_id",
                ctx_provider.get("provider_id"),
            ),
            provider_tax_id=self._resolve_optional_str(
                extracted_map, "provider_tax_id",
                ctx_provider.get("tax_id"),
            ),
        )

        # Persist the schema-validated summary.
        self._write_json(
            run_dir / "claim_summary.json",
            summary.model_dump(mode="json"),
        )

        audit_entries.append(
            f"Overall extraction confidence: {overall_confidence:.4f}."
        )
        if low_confidence_fields:
            audit_entries.append(
                f"{len(low_confidence_fields)} field(s) below the "
                f"{flag_threshold} confidence threshold."
            )
        for doc_type, filename in missing_required_docs:
            audit_entries.append(
                f"Missing required document: {doc_type} ({filename})."
            )
        audit_entries.append(
            "Manual review required."
            if requires_manual_review
            else "No manual review required."
        )
        audit_entries.append("Claim summary written for Agent C.")

        self._write_audit_log(
            run_dir / "audit_log_agent_b.md",
            claim_id,
            audit_entries,
        )

        result = summary.model_dump(mode="json")
        status, next_agent = self._determine_route(requires_manual_review)
        result["status"] = status
        result["next_agent"] = next_agent
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _missing_required_documents(
        bundle_path: Path,
        documents: list[dict[str, Any]],
    ) -> list[tuple[str, str]]:
        """
        Return (doc_type, filename) for every required document that cannot
        be processed: explicitly marked MISSING, or absent from disk.
        Reference-only optional evidence (e.g. photos) is ignored.

        A document is treated as a hard error when it is flagged
        ``status: MISSING`` (the data source omits the ``required`` key in
        that case), or when it is marked required yet cannot be read.
        """
        # Document types the pipeline cannot settle a claim without.
        core_types = {"fnol", "policy", "repair_estimate"}

        missing: list[tuple[str, str]] = []
        for document in documents:
            filename = str(document.get("file", ""))
            doc_type = str(document.get("type", "other"))
            status = str(document.get("status", "")).upper()
            required = bool(document.get("required", False)) or doc_type in core_types

            # Reference-only optional evidence (photos, medical scans) is fine.
            if status == "REFERENCE_ONLY" or document.get("status") == "reference_only":
                continue

            file_exists = (bundle_path / filename).is_file()
            processable = bool(document.get("processable", True))

            if status == "MISSING":
                missing.append((doc_type, filename))
            elif required and not processable and not file_exists:
                missing.append((doc_type, filename))
            elif required and processable and not file_exists:
                missing.append((doc_type, filename))

        return missing

    def _build_findings(
        self,
        claim_id: str,
        overall_confidence: float,
        min_auto: float,
        low_ocr: bool,
        low_confidence_fields: list[str],
        missing_required_docs: list[tuple[str, str]],
        created_at: str,
    ) -> list[Finding]:
        """
        Emit intake-quality findings. Agent B reports; Agent H routes.
        """
        findings: list[Finding] = []

        def add_finding(
            code: str,
            severity: Severity,
            recommendation: str,
            evidence_links: list[str],
            confidence: float,
        ) -> None:
            findings.append(
                Finding(
                    finding_id=f"B-{len(findings) + 1:03d}-{code}",
                    agent="agent_document_extraction",
                    severity=severity,
                    confidence=round(confidence, 4),
                    evidence_links=evidence_links,
                    recommendation=recommendation,
                    open_questions=[],
                    requires_human_review=(
                        severity in {Severity.critical, Severity.high}
                    ),
                    timestamp=created_at,
                )
            )

        for doc_type, filename in missing_required_docs:
            add_finding(
                code="MISSING_REQUIRED_DOCUMENT",
                severity=Severity.high,
                recommendation=(
                    f"Request the missing required document: {filename}"
                ),
                evidence_links=[f"manifest.yaml:documents:{doc_type}"],
                confidence=1.0,
            )

        if low_ocr or overall_confidence < min_auto:
            add_finding(
                code="LOW_OCR_CONFIDENCE",
                severity=Severity.high,
                recommendation=(
                    "Route to manual adjuster review — document quality is "
                    "below the auto-processing confidence threshold."
                ),
                evidence_links=[
                    f"claim_summary.json:overall_confidence={overall_confidence:.4f}"
                ],
                confidence=round(1.0 - overall_confidence, 4),
            )
        elif low_confidence_fields:
            # Informational only — not enough to halt auto-processing.
            add_finding(
                code="LOW_CONFIDENCE_FIELDS",
                severity=Severity.low,
                recommendation=(
                    "Spot-check the low-confidence fields before settlement."
                ),
                evidence_links=[
                    f"claim_summary.json:low_confidence_fields="
                    f"{','.join(low_confidence_fields)}"
                ],
                confidence=0.5,
            )

        return findings

    @staticmethod
    def _determine_route(requires_manual_review: bool) -> tuple[str, str]:
        """
        Agent B always forwards to Agent C. The manual-review signal travels
        with the claim summary; Agent H makes the terminal routing decision.
        """
        if requires_manual_review:
            return "ready_for_coverage_validation_with_review", "agent_c"
        return "ready_for_coverage_validation", "agent_c"

    @staticmethod
    def _resolve_str(
        extracted: dict[str, ExtractedField],
        field_name: str,
        fallback: Any,
    ) -> str:
        """Prefer a usable extracted value, else the context fallback."""
        ef = extracted.get(field_name)
        if ef is not None and ef.value not in (None, "", []):
            return str(ef.value)
        return str(fallback) if fallback is not None else ""

    @staticmethod
    def _resolve_optional_str(
        extracted: dict[str, ExtractedField],
        field_name: str,
        fallback: Any,
    ) -> str | None:
        """Like _resolve_str but preserves None when nothing is available."""
        ef = extracted.get(field_name)
        if ef is not None and ef.value not in (None, "", []):
            return str(ef.value)
        if fallback in (None, "", []):
            return None
        return str(fallback)

    @staticmethod
    def _resolve_float(
        extracted: dict[str, ExtractedField],
        field_name: str,
        fallback: Any,
    ) -> float:
        """Prefer a numeric extracted value, else parse/coerce the fallback."""
        ef = extracted.get(field_name)
        if ef is not None:
            value = ef.value
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str) and value.strip():
                try:
                    return float(value.replace(",", "").replace("$", "").strip())
                except ValueError:
                    pass
        try:
            return float(fallback)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        """Read a YAML file and return a dictionary."""
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        return data or {}

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        """Read a JSON file and return a dictionary."""
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        """Write readable, deterministic JSON output."""
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
            file.write("\n")

    @staticmethod
    def _write_audit_log(
        path: Path,
        claim_id: str,
        entries: list[str],
    ) -> None:
        """Write the human-readable Agent B audit trail."""
        lines = [f"# Audit Log (Agent B): {claim_id}", ""]
        lines.extend(f"- {entry}" for entry in entries)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
