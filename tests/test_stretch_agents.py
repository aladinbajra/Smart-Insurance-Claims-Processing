from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.pipeline.runner import PipelineRunner
from backend.tools.ocr import ocr_available, ocr_image, ocr_image_bytes
from backend.tools.tracker_poster import post_ticket, post_ticket_file


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dataset(scenario: str) -> Path:
    return _root() / "datasets" / "SICPS_Dataset (3)" / "SICPS_Dataset" / scenario


def _runner(tmp_path: Path) -> PipelineRunner:
    return PipelineRunner(
        config_path=_root() / "config" / "policy.yaml", runs_dir=tmp_path / "runs"
    )


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Liability Agent (7th, REQ-047)
# ---------------------------------------------------------------------------

def test_liability_agent_writes_split(tmp_path: Path) -> None:
    result = _runner(tmp_path).run(_dataset("scenario_02_total_loss"))
    lia = _load(result.run_dir / "liability_result.json")

    assert lia["claim_id"] == "CLM-2026-002"
    assert "rule_applied" in lia
    total = lia["insured_liability_pct"] + lia["third_party_liability_pct"]
    assert total in (0.0, 100.0)  # full allocation or no-fault


def test_liability_not_applicable_for_property_claim(tmp_path: Path) -> None:
    # Homeowners is a first-party property claim — no fault split.
    result = _runner(tmp_path).run(_dataset("scenario_09_homeowners"))
    lia = _load(result.run_dir / "liability_result.json")
    assert lia["rule_applied"] == "first_party_property_no_split"
    assert lia["at_fault_party"] == "not_applicable"


# ---------------------------------------------------------------------------
# Compliance Agent (8th, REQ-048)
# ---------------------------------------------------------------------------

def test_compliance_agent_passes_for_clean_config(tmp_path: Path) -> None:
    result = _runner(tmp_path).run(_dataset("scenario_01_clean_auto"))
    comp = _load(result.run_dir / "compliance_result.json")

    assert comp["framework"] == "NAIC"
    assert comp["compliant"] is True
    assert all(c["passed"] for c in comp["checks"])


# ---------------------------------------------------------------------------
# Anomaly Detection Agent (REQ-052)
# ---------------------------------------------------------------------------

def test_anomaly_total_loss_not_double_flagged(tmp_path: Path) -> None:
    # A recognized total loss is expected behaviour, not an anomaly.
    result = _runner(tmp_path).run(_dataset("scenario_02_total_loss"))
    anomaly = _load(result.run_dir / "anomaly_report.json")
    codes = [a["code"] for a in anomaly["anomalies"]]
    assert "REPAIR_EXCEEDS_VALUE" not in codes


def test_anomaly_flags_billing_variance(tmp_path: Path) -> None:
    result = _runner(tmp_path).run(_dataset("scenario_07_fraud_ring"))
    anomaly = _load(result.run_dir / "anomaly_report.json")
    codes = [a["code"] for a in anomaly["anomalies"]]
    assert "BILLING_VARIANCE_OUTLIER" in codes


# ---------------------------------------------------------------------------
# Agent H merges the stretch agents' findings; metrics lists them
# ---------------------------------------------------------------------------

def test_h_merges_stretch_findings(tmp_path: Path) -> None:
    result = _runner(tmp_path).run(_dataset("scenario_07_fraud_ring"))
    approval = _load(result.run_dir / "approval_packet.json")
    agents = {f["agent"] for f in approval["all_findings"]}
    assert {"agent_liability", "agent_compliance", "agent_anomaly"} <= agents


def test_metrics_lists_stretch_agents(tmp_path: Path) -> None:
    result = _runner(tmp_path).run(_dataset("scenario_07_fraud_ring"))
    metrics = _load(result.run_dir / "metrics.json")
    for agent in ("agent_liability", "agent_compliance", "agent_anomaly"):
        assert agent in metrics["agents_executed"]


# ---------------------------------------------------------------------------
# Determinism still holds with the new artifacts
# ---------------------------------------------------------------------------

def test_determinism_with_stretch_agents(tmp_path: Path) -> None:
    assert _runner(tmp_path).verify_determinism(_dataset("scenario_07_fraud_ring")) is True


# ---------------------------------------------------------------------------
# OCR (REQ-050) — graceful degradation, never raises
# ---------------------------------------------------------------------------

def test_ocr_graceful_degradation() -> None:
    assert isinstance(ocr_available(), bool)
    assert ocr_image("does_not_exist.png") == ("", 0.0)
    assert ocr_image_bytes(b"not-an-image") == ("", 0.0)


def test_ocr_reads_text_when_engine_available(tmp_path: Path) -> None:
    # Exercises the real OCR path on a generated image. If the tesseract engine
    # is installed, it must read the text; otherwise it degrades to ("", 0.0).
    Image = pytest.importorskip("PIL.Image")
    from PIL import ImageDraw

    img = Image.new("RGB", (260, 80), "white")
    ImageDraw.Draw(img).text((12, 28), "HELLO SICPS", fill="black")
    path = tmp_path / "scanned.png"
    img.save(path)

    text, conf = ocr_image(path)
    if ocr_available():
        assert "HELLO" in text.upper()
        assert conf > 0.0
    else:
        assert (text, conf) == ("", 0.0)


# ---------------------------------------------------------------------------
# Cross-claim duplicate detection (deterministic, VIN-based sibling scan)
# ---------------------------------------------------------------------------

def test_cross_claim_duplicate_detected(tmp_path: Path) -> None:
    # scenario_12 shares a VIN with the earlier scenario_07 -> flagged duplicate.
    result = _runner(tmp_path).run(_dataset("scenario_12_duplicate_claim"))
    approval = _load(result.run_dir / "approval_packet.json")
    dup = [f for f in approval["all_findings"] if "DUPLICATE" in f["finding_id"]]
    assert dup
    assert any("CLM-2026-007" in str(f["evidence_links"]) for f in dup)


def test_original_claim_not_flagged_duplicate(tmp_path: Path) -> None:
    # scenario_07 is the earliest claim on that VIN -> it is NOT a duplicate.
    result = _runner(tmp_path).run(_dataset("scenario_07_fraud_ring"))
    approval = _load(result.run_dir / "approval_packet.json")
    dup = [f for f in approval["all_findings"] if "DUPLICATE" in f["finding_id"]]
    assert dup == []


# ---------------------------------------------------------------------------
# Tracker posting (REQ-051) — graceful dry-run when no tracker is configured
# ---------------------------------------------------------------------------

def test_tracker_dry_run_without_config(monkeypatch) -> None:
    monkeypatch.delenv("TRACKER_TYPE", raising=False)
    result = post_ticket({"ticket_id": "SICPS-X", "title": "T", "claim_id": "X"})
    assert result["posted"] is False
    assert result["mode"] == "dry-run"


def test_tracker_post_from_run_writes_result(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TRACKER_TYPE", raising=False)
    run = _runner(tmp_path).run(_dataset("scenario_07_fraud_ring"))
    result = post_ticket_file(run.run_dir)
    assert result["ticket_id"] == "SICPS-CLM-2026-007"
    assert (run.run_dir / "tracker_post_result.json").is_file()


def test_tracker_jira_incomplete_env(monkeypatch) -> None:
    monkeypatch.setenv("TRACKER_TYPE", "jira")
    for var in ("JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN", "JIRA_PROJECT_KEY"):
        monkeypatch.delenv(var, raising=False)
    result = post_ticket({"ticket_id": "SICPS-X"})
    assert result["posted"] is False
    assert "incomplete" in result["reason"]


def test_tracker_ado_incomplete_env(monkeypatch) -> None:
    monkeypatch.setenv("TRACKER_TYPE", "ado")
    for var in ("ADO_ORG_URL", "ADO_PROJECT", "ADO_PAT"):
        monkeypatch.delenv(var, raising=False)
    result = post_ticket({"ticket_id": "SICPS-X"})
    assert result["posted"] is False
    assert "incomplete" in result["reason"]


# --- Mocked HTTP integration (no real Jira/ADO needed) ---

class _FakeResp:
    def __init__(self, status: int, data: dict | None = None, text: str = "") -> None:
        self.status_code = status
        self._data = data or {}
        self.text = text

    def json(self) -> dict:
        return self._data


class _FakeClient:
    def __init__(self, get_resp=None, post_resp=None) -> None:
        self._get, self._post = get_resp, post_resp

    def __enter__(self):
        return self

    def __exit__(self, *_a) -> bool:
        return False

    def get(self, *_a, **_k):
        return self._get

    def post(self, *_a, **_k):
        return self._post


def _jira_env(monkeypatch) -> None:
    monkeypatch.setenv("TRACKER_TYPE", "jira")
    monkeypatch.setenv("JIRA_BASE_URL", "https://x.atlassian.net")
    monkeypatch.setenv("JIRA_EMAIL", "a@b.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "tok")
    monkeypatch.setenv("JIRA_PROJECT_KEY", "SIC")


def test_tracker_jira_creates_issue(monkeypatch) -> None:
    import backend.tools.tracker_poster as tp

    _jira_env(monkeypatch)
    fake = _FakeClient(_FakeResp(200, {"issues": []}), _FakeResp(201, {"key": "SIC-1"}))
    monkeypatch.setattr(tp.httpx, "Client", lambda *a, **k: fake)
    result = post_ticket({"ticket_id": "SICPS-X", "title": "T", "claim_id": "X"})
    assert result["posted"] is True
    assert result["action"] == "created"
    assert result["issue_key"] == "SIC-1"


def test_tracker_jira_idempotent_existing(monkeypatch) -> None:
    import backend.tools.tracker_poster as tp

    _jira_env(monkeypatch)
    fake = _FakeClient(_FakeResp(200, {"issues": [{"key": "SIC-9"}]}), _FakeResp(201))
    monkeypatch.setattr(tp.httpx, "Client", lambda *a, **k: fake)
    result = post_ticket({"ticket_id": "SICPS-X", "title": "T"})
    assert result["posted"] is True
    assert result["action"] == "exists"
    assert result["issue_key"] == "SIC-9"


def test_tracker_ado_creates_work_item(monkeypatch) -> None:
    import backend.tools.tracker_poster as tp

    monkeypatch.setenv("TRACKER_TYPE", "ado")
    monkeypatch.setenv("ADO_ORG_URL", "https://dev.azure.com/org")
    monkeypatch.setenv("ADO_PROJECT", "proj")
    monkeypatch.setenv("ADO_PAT", "pat")
    fake = _FakeClient(post_resp=_FakeResp(201, {"id": 123}))
    monkeypatch.setattr(tp.httpx, "Client", lambda *a, **k: fake)
    result = post_ticket({"ticket_id": "SICPS-X", "title": "T"})
    assert result["posted"] is True
    assert result["work_item_id"] == 123
