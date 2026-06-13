from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from backend.models.schemas import (
    ContextPacket,
    EvidenceIndexEntry,
    Finding,
    ManifestSchema,
    Severity,
)

# Reuse the shared PDF extractor already created by the group.
# Agent A uses it to read raw FNOL and supporting-document text.
from backend.tools.pdf_extractor import extract_bundle, load_policy


class AgentFNOLIntake:
    """
    Agent A: FNOL Intake and Context Gatekeeper.

    This is the first agent in the SICPS pipeline.

    Responsibilities:
    - Read a submitted claim bundle.
    - Parse raw FNOL and supporting PDF text.
    - Validate the manifest using the shared SICPS schema.
    - Classify the claim type.
    - Build a universal evidence index.
    - Detect early intake risks.
    - Create the context packet used by downstream agents.
    - Write traceable JSON and Markdown run artifacts.
    """

    def __init__(
        self,
        config_path: str | Path,
        runs_dir: str | Path,
    ) -> None:
        # Store the shared policy configuration path.
        self.config_path = Path(config_path)

        # Store the folder where Agent A writes generated artifacts.
        self.runs_dir = Path(runs_dir)

        # Load the YAML configuration for simple threshold access.
        self.policy = self._load_yaml(self.config_path)

        # Load the validated policy object required by the shared PDF extractor.
        self.policy_pack = load_policy(self.config_path)

    def process(self, bundle_dir: str | Path) -> dict[str, Any]:
        """
        Process one SICPS claim bundle.

        Agent A creates:
        - runs/<claim_id>/context_packet.json
        - runs/<claim_id>/evidence_index.json
        - runs/<claim_id>/audit_log.md
        """

        bundle_path = Path(bundle_dir)
        manifest_path = bundle_path / "manifest.yaml"

        # A valid claims bundle must contain a manifest file.
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest file not found: {manifest_path}"
            )

        # Read the original YAML content.
        raw_manifest = self._load_yaml(manifest_path)

        # The synthetic dataset lists some photos as reference-only evidence.
        # Their physical files are intentionally not included.
        # Add an in-memory confidence value so schema validation succeeds
        # without modifying the shared dataset or the shared schema.
        self._normalize_reference_only_documents(raw_manifest)

        # Validate the submitted bundle against the group's shared contract.
        manifest = ManifestSchema.model_validate(raw_manifest)

        # Extract structured data directly from the submitted PDF files.
        # This uses the shared extractor instead of creating a second parser.
        extracted_fields = extract_bundle(
            claims_dir=bundle_path,
            policy=self.policy_pack,
        )

        audit_data = raw_manifest.get("audit", {}) or {}

        # Reuse the idempotency key when available.
        # This keeps repeated runs deterministic.
        run_id = str(
            audit_data.get(
                "idempotency_key",
                manifest.claim_id,
            )
        )

        # Reuse the dataset timestamp for traceability.
        created_at = str(
            audit_data.get(
                "created_at",
                "1970-01-01T00:00:00Z",
            )
        )

        # Reuse the provided hash when available.
        # Otherwise calculate a deterministic hash from the claim bundle.
        input_hash = str(
            audit_data.get(
                "input_hash",
                self._calculate_bundle_hash(bundle_path),
            )
        )

        # Create an isolated run folder for this claim.
        run_dir = self.runs_dir / manifest.claim_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Keep a readable record of Agent A actions.
        audit_entries: list[str] = [
            "Agent A started processing the claims bundle.",
            "Manifest loaded successfully.",
            "Manifest validated against the shared SICPS schema.",
            (
                f"Raw PDF extraction completed with "
                f"{len(extracted_fields)} structured field(s)."
            ),
        ]

        # Build the evidence index required by downstream agents.
        (
            evidence_index,
            missing_required_files,
            missing_processable_files,
        ) = self._build_evidence_index(
            bundle_path=bundle_path,
            raw_documents=raw_manifest.get("documents", []) or [],
            audit_entries=audit_entries,
        )

        # Detect intake risks and raw-document inconsistencies.
        findings = self._build_intake_findings(
            manifest=manifest,
            extracted_fields=extracted_fields,
            missing_required_files=missing_required_files,
            missing_processable_files=missing_processable_files,
            created_at=created_at,
        )

        if findings:
            audit_entries.append(
                f"Agent A recorded {len(findings)} intake finding(s)."
            )
        else:
            audit_entries.append(
                "No Agent A intake findings detected."
            )

        # Create the shared context structure consumed by later agents.
        context_packet = ContextPacket(
            claim_id=manifest.claim_id,
            run_id=run_id,
            claim_type=manifest.claim_type,
            priority=manifest.priority,
            claimant=manifest.claimant,
            policy=manifest.policy,
            incident=manifest.incident,
            vehicle=manifest.vehicle,
            financials=manifest.financials,
            provider=manifest.provider,
            flags=manifest.flags,
            evidence_index=evidence_index,
            findings=findings,
            input_hash=input_hash,
            created_at=created_at,
        )

        # Save the standardized context packet.
        self._write_json(
            run_dir / "context_packet.json",
            context_packet.model_dump(mode="json"),
        )

        # Save the universal evidence index separately for auditability.
        self._write_json(
            run_dir / "evidence_index.json",
            {
                "claim_id": manifest.claim_id,
                "run_id": run_id,
                "evidence_index": [
                    item.model_dump(mode="json")
                    for item in evidence_index
                ],
            },
        )

        audit_entries.append(
            f"Evidence index created with {len(evidence_index)} item(s)."
        )

        audit_entries.append(
            "Context packet written for downstream agents."
        )

        # Save the human-readable audit log.
        self._write_audit_log(
            run_dir / "audit_log.md",
            manifest.claim_id,
            audit_entries,
        )

        # Keep the saved context packet strictly compatible with the
        # shared schema. Add terminal-display values only to the returned
        # dictionary so backend/main.py can print the routing result.
        result = context_packet.model_dump(mode="json")

        status, next_agent = self._determine_route(findings)

        result["status"] = status
        result["next_agent"] = next_agent

        return result

    @staticmethod
    def _normalize_reference_only_documents(
        raw_manifest: dict[str, Any],
    ) -> None:
        """
        Add an in-memory confidence value for reference-only evidence.

        Some synthetic photo entries intentionally do not include physical
        files or OCR confidence values.
        """

        for document in raw_manifest.get("documents", []):
            if (
                document.get("processable") is False
                and "expected_confidence" not in document
            ):
                document["expected_confidence"] = "not_applicable"

    def _build_evidence_index(
        self,
        bundle_path: Path,
        raw_documents: list[dict[str, Any]],
        audit_entries: list[str],
    ) -> tuple[
        list[EvidenceIndexEntry],
        list[str],
        list[str],
    ]:
        """
        Build the universal evidence index.

        Required processable files must exist physically.
        Reference-only evidence remains indexed even when its physical
        file is intentionally omitted from the synthetic dataset.
        """

        evidence_index: list[EvidenceIndexEntry] = []
        missing_required_files: list[str] = []
        missing_processable_files: list[str] = []

        for document in raw_documents:
            doc_type = str(
                document.get("type", "other")
            )

            relative_path = str(
                document.get("file", "")
            )

            processable = bool(
                document.get("processable", False)
            )

            required = bool(
                document.get("required", False)
            )

            file_path = bundle_path / relative_path
            file_exists = file_path.is_file()

            # Required files must exist before processing can continue.
            if required and not file_exists:
                missing_required_files.append(relative_path)

                audit_entries.append(
                    f"Missing required file detected: {relative_path}."
                )

            # A processable file should exist even when it is optional.
            elif processable and not file_exists:
                missing_processable_files.append(relative_path)

                audit_entries.append(
                    f"Missing processable file detected: {relative_path}."
                )

            # Reference-only photos are valid evidence-index entries.
            if not processable and not file_exists:
                audit_entries.append(
                    f"Reference-only evidence indexed: {relative_path}."
                )

            evidence_index.append(
                EvidenceIndexEntry(
                    doc_type=doc_type,
                    file=relative_path,
                    provides=self._fields_provided_by(doc_type),
                    processable=processable and file_exists,
                    page_count=1,
                )
            )

        return (
            evidence_index,
            missing_required_files,
            missing_processable_files,
        )

    @staticmethod
    def _fields_provided_by(doc_type: str) -> list[str]:
        """
        Describe the claim data provided by each evidence type.
        """

        field_map = {
            "fnol": [
                "claimant",
                "incident",
                "claim_type",
            ],
            "policy": [
                "policy",
                "coverage",
            ],
            "repair_estimate": [
                "financials",
                "repair_estimate",
            ],
            "police_report": [
                "incident",
                "police_report",
            ],
            "vendor_master": [
                "provider",
            ],
            "photo": [
                "damage_evidence",
            ],
        }

        return field_map.get(
            doc_type,
            ["supporting_evidence"],
        )

    def _build_intake_findings(
        self,
        manifest: ManifestSchema,
        extracted_fields: dict[str, Any],
        missing_required_files: list[str],
        missing_processable_files: list[str],
        created_at: str,
    ) -> list[Finding]:
        """
        Detect early intake risks.

        Agent A performs initial filtering only.
        Detailed extraction remains the responsibility of Agent B.
        Detailed fraud scoring remains the responsibility of Agent E.
        """

        findings: list[Finding] = []

        def add_finding(
            code: str,
            severity: Severity,
            recommendation: str,
            evidence_links: list[str],
        ) -> None:
            findings.append(
                Finding(
                    finding_id=(
                        f"A-{len(findings) + 1:03d}-{code}"
                    ),
                    agent="agent_fnol_intake",
                    severity=severity,
                    confidence=1.0,
                    evidence_links=evidence_links,
                    recommendation=recommendation,
                    open_questions=[],
                    requires_human_review=(
                        severity in {
                            Severity.critical,
                            Severity.high,
                        }
                    ),
                    timestamp=created_at,
                )
            )

        # Stop claims that are missing mandatory evidence.
        for filename in missing_required_files:
            add_finding(
                code="MISSING_REQUIRED_DOCUMENT",
                severity=Severity.critical,
                recommendation=(
                    f"Request the missing required document: {filename}"
                ),
                evidence_links=[filename],
            )

        # Isolate claims with absent files that should be processable.
        for filename in missing_processable_files:
            add_finding(
                code="MISSING_PROCESSABLE_DOCUMENT",
                severity=Severity.high,
                recommendation=(
                    f"Review the missing processable document: {filename}"
                ),
                evidence_links=[filename],
            )

        # Confirm that raw PDF extraction produced the minimum fields
        # needed for FNOL classification and policy mapping.
        extracted_claim_type = self._extracted_value(
            extracted_fields,
            "claim_type",
        )

        extracted_policy_number = self._extracted_value(
            extracted_fields,
            "policy_number",
        )

        if not extracted_claim_type:
            add_finding(
                code="FNOL_CLAIM_TYPE_NOT_EXTRACTED",
                severity=Severity.high,
                recommendation=(
                    "Isolate the claim for manual FNOL classification."
                ),
                evidence_links=["fnol.pdf:claim_type"],
            )

        elif str(extracted_claim_type).lower() != manifest.claim_type.lower():
            add_finding(
                code="CLAIM_TYPE_MISMATCH",
                severity=Severity.high,
                recommendation=(
                    "Review the claim-type mismatch between "
                    "the raw FNOL and the manifest."
                ),
                evidence_links=[
                    "fnol.pdf:claim_type",
                    "manifest.yaml:claim_type",
                ],
            )

        if not extracted_policy_number:
            add_finding(
                code="POLICY_NUMBER_NOT_EXTRACTED",
                severity=Severity.high,
                recommendation=(
                    "Isolate the claim for manual policy matching."
                ),
                evidence_links=["fnol.pdf:policy_number"],
            )

        elif (
            str(extracted_policy_number)
            != manifest.policy.policy_number
        ):
            add_finding(
                code="POLICY_NUMBER_MISMATCH",
                severity=Severity.high,
                recommendation=(
                    "Review the policy-number mismatch between "
                    "the raw documents and the manifest."
                ),
                evidence_links=[
                    "fnol.pdf:policy_number",
                    "manifest.yaml:policy.policy_number",
                ],
            )

        # Record an inactive or expired policy as an early risk signal.
        # Agent C remains responsible for full coverage validation.
        if (
            not manifest.policy.is_active
            or manifest.policy.policy_expired
        ):
            add_finding(
                code="INACTIVE_OR_EXPIRED_POLICY",
                severity=Severity.high,
                recommendation=(
                    "Isolate the claim for coverage review."
                ),
                evidence_links=[
                    "manifest.yaml:policy.is_active",
                    "manifest.yaml:policy.policy_expired",
                ],
            )

        # Record claimant-history risk signals.
        if manifest.claimant.new_claimant:
            add_finding(
                code="NEW_CLAIMANT",
                severity=Severity.medium,
                recommendation=(
                    "Continue processing with additional claimant review."
                ),
                evidence_links=[
                    "manifest.yaml:claimant.new_claimant"
                ],
            )

        if manifest.claimant.fraud_history:
            add_finding(
                code="PRIOR_FRAUD_HISTORY",
                severity=Severity.high,
                recommendation=(
                    "Isolate the claim and forward the fraud-history "
                    "signal to the fraud-detection agent."
                ),
                evidence_links=[
                    "manifest.yaml:claimant.fraud_history"
                ],
            )

        prior_claims_threshold = int(
            self.policy.get(
                "fraud",
                {},
            ).get(
                "prior_claims_siu_trigger",
                4,
            )
        )

        if (
            manifest.claimant.prior_claims_18mo
            >= prior_claims_threshold
        ):
            add_finding(
                code="MULTIPLE_PRIOR_CLAIMS",
                severity=Severity.medium,
                recommendation=(
                    "Forward the prior-claims signal "
                    "to the fraud-detection agent."
                ),
                evidence_links=[
                    "manifest.yaml:claimant.prior_claims_18mo"
                ],
            )

        # Record provider risk matches before downstream processing.
        if (
            manifest.provider is not None
            and manifest.provider.is_blacklisted
        ):
            add_finding(
                code="BLACKLISTED_PROVIDER",
                severity=Severity.high,
                recommendation=(
                    "Isolate the claim and forward the provider match "
                    "to the fraud-detection agent."
                ),
                evidence_links=[
                    "manifest.yaml:provider.is_blacklisted"
                ],
            )

        if (
            manifest.provider is not None
            and manifest.provider.fraud_ring_id is not None
        ):
            add_finding(
                code="KNOWN_FRAUD_RING_MATCH",
                severity=Severity.high,
                recommendation=(
                    "Isolate the claim and forward the fraud-ring "
                    "match to the fraud-detection agent."
                ),
                evidence_links=[
                    "manifest.yaml:provider.fraud_ring_id"
                ],
            )

        # Record catastrophe-event signals for special routing.
        if manifest.incident.cat_event:
            add_finding(
                code="CATASTROPHE_EVENT",
                severity=Severity.medium,
                recommendation=(
                    "Process the claim using catastrophe-event rules."
                ),
                evidence_links=[
                    "manifest.yaml:incident.cat_event"
                ],
            )

        # Detect duplicate or incomplete submissions.
        if manifest.flags.duplicate_claim:
            add_finding(
                code="POSSIBLE_DUPLICATE_CLAIM",
                severity=Severity.high,
                recommendation=(
                    "Isolate the claim and request duplicate-claim review."
                ),
                evidence_links=[
                    "manifest.yaml:flags.duplicate_claim"
                ],
            )

        if manifest.flags.missing_documents:
            add_finding(
                code="MISSING_DOCUMENTS_FLAG",
                severity=Severity.high,
                recommendation=(
                    "Review the submitted evidence before continuing."
                ),
                evidence_links=[
                    "manifest.yaml:flags.missing_documents"
                ],
            )

        return findings

    @staticmethod
    def _extracted_value(
        extracted_fields: dict[str, Any],
        field_name: str,
    ) -> Any:
        """
        Return a value extracted from the submitted PDFs.

        Return None when the field was not found or has no usable value.
        """

        extracted_field = extracted_fields.get(field_name)

        if extracted_field is None:
            return None

        value = extracted_field.value

        if value in ("", None):
            return None

        return value

    @staticmethod
    def _determine_route(
        findings: list[Finding],
    ) -> tuple[str, str | None]:
        """
        Decide whether a claim may continue to Agent B.

        Critical or high-risk findings isolate the claim immediately.
        Medium and low findings remain visible but allow processing.
        """

        requires_intake_review = any(
            finding.severity in {
                Severity.critical,
                Severity.high,
            }
            for finding in findings
        )

        if requires_intake_review:
            return "intake_review_required", None

        if findings:
            return (
                "ready_for_document_extraction_with_warnings",
                "agent_b",
            )

        return "ready_for_document_extraction", "agent_b"

    @staticmethod
    def _calculate_bundle_hash(
        bundle_path: Path,
    ) -> str:
        """
        Calculate a deterministic SHA-256 hash for the claim bundle.

        Re-running the same input produces the same hash.
        """

        digest = hashlib.sha256()

        for file_path in sorted(
            path
            for path in bundle_path.rglob("*")
            if path.is_file()
        ):
            relative_path = file_path.relative_to(bundle_path)

            digest.update(
                str(relative_path).encode("utf-8")
            )

            digest.update(
                file_path.read_bytes()
            )

        return f"sha256:{digest.hexdigest()}"

    @staticmethod
    def _load_yaml(
        path: Path,
    ) -> dict[str, Any]:
        """Read a YAML file and return a dictionary."""

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        return data or {}

    @staticmethod
    def _write_json(
        path: Path,
        data: dict[str, Any],
    ) -> None:
        """
        Write readable and deterministic JSON output.

        Sorted keys make repeated runs easier to compare.
        """

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=2,
                sort_keys=True,
            )

            file.write("\n")

    @staticmethod
    def _write_audit_log(
        path: Path,
        claim_id: str,
        entries: list[str],
    ) -> None:
        """Write the human-readable Agent A audit trail."""

        lines = [
            f"# Audit Log: {claim_id}",
            "",
        ]

        lines.extend(
            f"- {entry}"
            for entry in entries
        )

        path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )