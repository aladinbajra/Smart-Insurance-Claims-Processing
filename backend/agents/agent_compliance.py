from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from backend.models.schemas import Finding, Severity


class AgentCompliance:
    """
    Agent G (8th agent, stretch REQ-048): Regulatory-Compliance check.

    Deterministic NAIC / state compliance gate run before the final decision.
    It verifies that the run satisfies the regulatory and privacy rules defined
    in the policy pack (jurisdiction, PII redaction, data retention) and that
    coverage decisions carry a documented reason. Rule-based — no LLM (REQ-044).

    Output: runs/<claim_id>/compliance_result.json
    """

    # NAIC unfair-claims guidance: retention of at least ~6 years is typical.
    # Default only; the real threshold is read from policy.yaml (REQ-058).
    _DEFAULT_MIN_RETENTION_DAYS = 2190

    def __init__(self, config_path: str | Path, runs_dir: str | Path) -> None:
        self.runs_dir = Path(runs_dir)
        cfg = self._load_yaml(Path(config_path))
        self.compliance_cfg = cfg.get("compliance", {}) or {}
        self.privacy_cfg = cfg.get("privacy", {}) or {}
        self.min_retention_days = int(
            self.compliance_cfg.get(
                "min_data_retention_days", self._DEFAULT_MIN_RETENTION_DAYS
            )
        )

    def process(self, run_dir: str | Path) -> dict[str, Any]:
        run_path = Path(run_dir)
        context = self._load_json(run_path / "context_packet.json")
        coverage = self._load_json(run_path / "coverage_result.json")

        claim_id = str(context.get("claim_id", run_path.name))
        created_at = str(context.get("created_at", "1970-01-01T00:00:00Z"))
        policy = context.get("policy", {}) or {}
        state = str(policy.get("state", "")) or "US"

        framework = str(self.compliance_cfg.get("regulatory_framework", "NAIC"))
        jurisdiction = str(self.compliance_cfg.get("jurisdiction", "US"))
        retention = int(self.privacy_cfg.get("data_retention_days", 0))
        pii_redaction = bool(self.privacy_cfg.get("audit_log_pii_redaction", False))

        checks: list[dict[str, Any]] = []
        findings: list[Finding] = []

        def check(name: str, passed: bool, detail: str) -> None:
            checks.append({"check": name, "passed": passed, "detail": detail})

        check("regulatory_framework_defined", bool(framework), f"framework={framework}")
        check("jurisdiction_defined", bool(jurisdiction), f"jurisdiction={jurisdiction}")

        pii_ok = pii_redaction
        check("pii_redaction_enabled", pii_ok, f"audit_log_pii_redaction={pii_redaction}")
        if not pii_ok:
            findings.append(self._finding(
                "G-001-PII_REDACTION_DISABLED", Severity.high,
                "PII redaction is disabled; enable it to meet privacy compliance.",
                ["policy.yaml:privacy.audit_log_pii_redaction"], created_at,
                ["Confirm whether PII may be stored unmasked under jurisdiction rules."],
            ))

        retention_ok = retention >= self.min_retention_days
        check("data_retention_adequate", retention_ok,
              f"retention_days={retention} (min {self.min_retention_days})")
        if not retention_ok:
            findings.append(self._finding(
                "G-002-RETENTION_TOO_SHORT", Severity.medium,
                f"Data retention {retention}d is below the {self.min_retention_days}d minimum.",
                ["policy.yaml:privacy.data_retention_days"], created_at, [],
            ))

        # A coverage denial must carry a documented reason (NAIC unfair-claims).
        denial_documented = True
        if coverage and coverage.get("coverage_active") is False:
            denial_documented = bool(coverage.get("denial_reason"))
            check("denial_reason_documented", denial_documented,
                  f"denial_reason={coverage.get('denial_reason')!r}")
            if not denial_documented:
                findings.append(self._finding(
                    "G-003-DENIAL_REASON_MISSING", Severity.high,
                    "Coverage denied without a documented reason — compliance risk.",
                    ["coverage_result.json:denial_reason"], created_at, [],
                ))

        compliant = all(c["passed"] for c in checks)
        if compliant:
            findings.append(self._finding(
                "G-000-COMPLIANT", Severity.info,
                f"Claim meets {framework} / {jurisdiction} compliance checks.",
                ["policy.yaml:compliance"], created_at, [],
            ))

        result = {
            "claim_id": claim_id,
            "framework": framework,
            "jurisdiction": jurisdiction,
            "state": state,
            "compliant": compliant,
            "checks": checks,
            "findings": [f.model_dump(mode="json") for f in findings],
        }
        self._write_json(run_path / "compliance_result.json", result)
        return result

    @staticmethod
    def _finding(
        fid: str, severity: Severity, recommendation: str,
        evidence: list[str], created_at: str, open_q: list[str],
    ) -> Finding:
        return Finding(
            finding_id=fid,
            agent="agent_compliance",
            severity=severity,
            confidence=1.0,
            evidence_links=evidence,
            recommendation=recommendation,
            open_questions=open_q,
            requires_human_review=severity in {Severity.critical, Severity.high},
            timestamp=created_at,
        )

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
            file.write("\n")
