from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.models.schemas import Finding, Severity


class AgentLiability:
    """
    Agent F (7th agent, stretch REQ-047): Liability Determination.

    Deterministic, rule-based fault allocation by jurisdiction. It reads the
    context packet (incident facts) and produces a liability split between the
    insured and the third party, plus the rule that was applied. Rule-based —
    no LLM — so the result is reproducible (REQ-044).

    Output: runs/<claim_id>/liability_result.json
    """

    def __init__(self, runs_dir: str | Path) -> None:
        self.runs_dir = Path(runs_dir)

    def process(self, run_dir: str | Path) -> dict[str, Any]:
        run_path = Path(run_dir)
        context = self._load(run_path / "context_packet.json")

        claim_id = str(context.get("claim_id", run_path.name))
        created_at = str(context.get("created_at", "1970-01-01T00:00:00Z"))
        incident = context.get("incident", {}) or {}
        policy = context.get("policy", {}) or {}
        claim_type = str(context.get("claim_type", "")).lower()
        vehicle = context.get("vehicle") or {}
        has_real_vehicle = bool(vehicle.get("vin") or vehicle.get("make"))

        itype = str(incident.get("type", "")).lower()
        cat_event = bool(incident.get("cat_event", False))
        state = str(policy.get("state", "")) or "US"

        # Fault allocation only applies to third-party (auto) claims. First-party
        # property claims (e.g. homeowners, with no real vehicle) have no split.
        if claim_type and claim_type != "auto" and not has_real_vehicle:
            insured_pct, third_party_pct, at_fault, rule = (
                0.0, 0.0, "not_applicable", "first_party_property_no_split",
            )
        else:
            insured_pct, third_party_pct, at_fault, rule = self._allocate(itype, cat_event)

        # Comparative-negligence note (most US states); informational.
        jurisdiction_rule = f"{state}: comparative negligence"

        findings: list[Finding] = []
        undetermined = rule == "undetermined"
        findings.append(
            Finding(
                finding_id="F-001-LIABILITY_ASSESSED",
                agent="agent_liability",
                severity=Severity.medium if undetermined else Severity.info,
                confidence=0.6 if undetermined else 0.9,
                evidence_links=[
                    "context_packet.json:incident.type",
                    "context_packet.json:incident.cat_event",
                ],
                recommendation=(
                    f"Liability rule '{rule}': insured {insured_pct}% / "
                    f"third party {third_party_pct}% ({at_fault})."
                ),
                open_questions=(
                    ["Adjuster to confirm fault — incident type is inconclusive."]
                    if undetermined
                    else []
                ),
                requires_human_review=undetermined,
                timestamp=created_at,
            )
        )

        result = {
            "claim_id": claim_id,
            "at_fault_party": at_fault,
            "insured_liability_pct": insured_pct,
            "third_party_liability_pct": third_party_pct,
            "rule_applied": rule,
            "jurisdiction": state,
            "jurisdiction_rule": jurisdiction_rule,
            "requires_adjuster": undetermined,
            "findings": [f.model_dump(mode="json") for f in findings],
        }
        self._write_json(run_path / "liability_result.json", result)
        return result

    @staticmethod
    def _allocate(itype: str, cat_event: bool) -> tuple[float, float, str, str]:
        """Deterministic fault allocation (insured%, third_party%, party, rule)."""
        if cat_event or any(k in itype for k in ("hail", "storm", "weather", "flood", "catastrophe")):
            return 0.0, 0.0, "no_fault_act_of_nature", "act_of_nature"
        if any(k in itype for k in ("theft", "vandal")):
            return 0.0, 0.0, "no_fault_event", "no_fault_event"
        if "rear" in itype:
            return 0.0, 100.0, "third_party", "rear_end_third_party_at_fault"
        if any(k in itype for k in ("multi", "pileup", "pile-up")):
            return 50.0, 50.0, "shared", "multi_vehicle_shared_fault"
        if any(k in itype for k in ("single", "fixed", "object", "rollover")):
            return 100.0, 0.0, "insured", "single_vehicle_insured_at_fault"
        return 50.0, 50.0, "undetermined", "undetermined"

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
            file.write("\n")
