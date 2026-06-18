"""
Agent G (advisory): LLM Liability & Regulatory-Compliance assistant.
REQ-047 (Liability) + REQ-048 (Regulatory Compliance) — stretch goals.

This agent is ADVISORY ONLY and lives OUTSIDE the deterministic decision path.
It reads the run artifacts produced by the rule-based pipeline (A-E, H) and asks
an OpenAI model for a liability split and a NAIC compliance read. Its output is
written to runs/<claim_id>/llm_advisory.json + llm_advisory.md and never feeds
back into routing, settlement, or the determinism-checked artifacts.

Graceful degradation (REQ design constraint): with no OPENAI_API_KEY, no openai
package, or any API error, it writes a clearly-marked "unavailable" advisory and
returns — it never raises and never blocks the pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_SYSTEM_PROMPT = (
    "You are an insurance claims adjuster assistant. Given structured claim "
    "facts, produce a concise ADVISORY assessment. You do not make the final "
    "decision — the deterministic engine does. Respond ONLY with JSON matching "
    "this schema: {"
    '"liability": {"at_fault_party": str, "claimant_liability_pct": number, '
    '"insured_liability_pct": number, "rationale": str}, '
    '"compliance": {"framework": str, "compliant": bool, "concerns": [str], '
    '"required_disclosures": [str]}, '
    '"narrative": str, "confidence": number}. '
    "claimant_liability_pct + insured_liability_pct must sum to 100."
)


class AgentLLMAdvisory:
    """Advisory liability + compliance assessor backed by OpenAI."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def process(self, run_dir: str | Path) -> dict[str, Any]:
        run_path = Path(run_dir)
        context = self._load(run_path / "context_packet.json")
        coverage = self._load(run_path / "coverage_result.json")
        settlement = self._load(run_path / "settlement_calc.json")
        fraud = self._load(run_path / "fraud_score.json")
        approval = self._load(run_path / "approval_packet.json")

        claim_id = str(context.get("claim_id", run_path.name))

        facts = self._build_facts(
            claim_id, context, coverage, settlement, fraud, approval
        )

        advisory = self._call_openai(facts)
        advisory["claim_id"] = claim_id

        self._write_json(run_path / "llm_advisory.json", advisory)
        self._write_md(run_path / "llm_advisory.md", claim_id, advisory)
        return advisory

    # ------------------------------------------------------------------
    # OpenAI call (with graceful degradation)
    # ------------------------------------------------------------------

    def _call_openai(self, facts: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            return self._unavailable("OPENAI_API_KEY not set")

        try:
            from openai import OpenAI
        except ImportError:
            return self._unavailable("openai package not installed")

        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(facts, sort_keys=True)},
                ],
            )
            content = response.choices[0].message.content or "{}"
            advisory = json.loads(content)
            advisory["status"] = "ok"
            advisory["model"] = self.model
            advisory["advisory_only"] = True
            return advisory
        except Exception as exc:  # noqa: BLE001 — advisory must never crash the run
            return self._unavailable(f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _unavailable(reason: str) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "reason": reason,
            "advisory_only": True,
            "liability": None,
            "compliance": None,
            "narrative": (
                "LLM advisory unavailable; deterministic engine decision stands. "
                f"({reason})"
            ),
            "confidence": 0.0,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_facts(
        claim_id: str,
        context: dict,
        coverage: dict,
        settlement: dict,
        fraud: dict,
        approval: dict,
    ) -> dict[str, Any]:
        incident = context.get("incident", {}) or {}
        policy = context.get("policy", {}) or {}
        return {
            "claim_id": claim_id,
            "claim_type": context.get("claim_type"),
            "jurisdiction_state": policy.get("state"),
            "incident": {
                "type": incident.get("type"),
                "severity": incident.get("severity"),
                "location": incident.get("location"),
                "police_report_filed": incident.get("police_report_filed"),
                "witness_count": incident.get("witness_count"),
                "cat_event": incident.get("cat_event"),
            },
            "coverage_active": coverage.get("coverage_active"),
            "denial_reason": coverage.get("denial_reason"),
            "late_notice_violation": coverage.get("late_notice_violation"),
            "gross_amount": settlement.get("gross_amount"),
            "net_settlement": settlement.get("net_settlement"),
            "total_loss_flag": settlement.get("total_loss_flag"),
            "fraud_score": fraud.get("fraud_score"),
            "risk_level": fraud.get("risk_level"),
            "siu_referral": fraud.get("siu_referral"),
            "engine_routing_decision": approval.get("routing_decision"),
        }

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
            file.write("\n")

    @staticmethod
    def _write_md(path: Path, claim_id: str, advisory: dict[str, Any]) -> None:
        lines = [
            f"# LLM Advisory (Liability & Compliance): {claim_id}",
            "",
            "> Advisory only — does not change the deterministic engine decision.",
            "",
            f"- Status: **{advisory.get('status')}**",
            f"- Model: {advisory.get('model', 'n/a')}",
            f"- Confidence: {advisory.get('confidence', 0.0)}",
            "",
        ]
        liability = advisory.get("liability") or {}
        if liability:
            lines += [
                "## Liability",
                f"- At-fault party: {liability.get('at_fault_party')}",
                f"- Claimant liability: {liability.get('claimant_liability_pct')}%",
                f"- Insured liability: {liability.get('insured_liability_pct')}%",
                f"- Rationale: {liability.get('rationale')}",
                "",
            ]
        compliance = advisory.get("compliance") or {}
        if compliance:
            concerns = ", ".join(compliance.get("concerns", []) or []) or "none"
            disclosures = ", ".join(
                compliance.get("required_disclosures", []) or []
            ) or "none"
            lines += [
                "## Compliance",
                f"- Framework: {compliance.get('framework')}",
                f"- Compliant: {compliance.get('compliant')}",
                f"- Concerns: {concerns}",
                f"- Required disclosures: {disclosures}",
                "",
            ]
        lines += ["## Narrative", advisory.get("narrative", ""), ""]
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m backend.agents.agent_llm_advisory <run_dir>"""
    import argparse

    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run the advisory LLM liability/compliance agent on a run dir."
    )
    parser.add_argument("run_dir", help="Path to runs/<claim_id>/")
    args = parser.parse_args(argv)

    result = AgentLLMAdvisory().process(args.run_dir)
    print(f"Advisory status: {result.get('status')} ({result.get('model', 'n/a')})")
    if result.get("liability"):
        lia = result["liability"]
        print(
            f"  Liability -> insured {lia.get('insured_liability_pct')}% / "
            f"claimant {lia.get('claimant_liability_pct')}%"
        )
    print(f"  artifacts: {Path(args.run_dir) / 'llm_advisory.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
