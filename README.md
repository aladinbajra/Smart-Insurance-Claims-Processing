# Smart Insurance Claims Processing System (SICPS)

A **deterministic, multi-agent AI pipeline** that automates insurance claims
validation and settlement. A claim goes in; the system extracts the data,
validates coverage, calculates the settlement, screens for fraud, and routes it
to one of nine decisions — with a complete, auditable trail.

Built with Python, FastAPI, and Streamlit, with an optional OpenAI advisory agent.

---

## Highlights

- **Multi-agent pipeline** — specialized agents, each with one job, chained A → H.
- **Deterministic & file-based** — agents communicate only through files in a shared
  run directory; the same input yields **byte-for-byte identical** output.
- **Idempotent** — a SHA-256 cache skips unchanged claims.
- **Dynamic policy pack** — all thresholds live in `config/policy.yaml`; change a value,
  re-run, and decisions change with **no code edits**.
- **Auditable & compliant** — every decision carries evidence pointers (incl. PDF
  bounding boxes), with PII redaction (NAIC-aware).
- **Matching precision** — invoice ↔ PO ↔ GRN three-way checking.
- **CoreLogic-ready output** — settlement payloads in JSON, CSV, and Markdown.
- **Two input modes** — a full claims bundle, or a single FNOL document.

---

## The Agents

### Core pipeline (A → H)

| Agent | File | Responsibility |
| ----- | ---- | -------------- |
| **A** | `agent_fnol_intake.py` | FNOL intake & classification, context packet, evidence index, risk + cross-claim duplicate detection |
| **B** | `agent_document_extraction.py` | Document extraction — fields, confidence scores, bounding boxes, synthetic fallback |
| **C** | `agent_coverage_validation.py` | Coverage validation — expiry, exclusions, deductibles, late notice (+ state overrides), sub-limits |
| **D** | `agent_damage_assessment.py` | Damage assessment — gross/net, total-loss, sub-limit cap, invoice/PO/GRN matching |
| **E** | `agent_fraud_detection.py` | Fraud detection — blacklist, rings, staged accidents, provider collusion, SIU referral |
| **H** | `agent_exception_triage.py` | Orchestrator & judge — merges findings, makes the routing decision, writes final artifacts |

### Stretch agents (advisory, outside the deterministic decision)

| Agent | File | Responsibility |
| ----- | ---- | -------------- |
| Liability | `agent_liability.py` | Jurisdiction-based fault allocation |
| Compliance | `agent_compliance.py` | NAIC / state compliance checks |
| Anomaly | `agent_anomaly.py` | Outlier + surge detection |
| LLM Advisory | `agent_llm_advisory.py` | OpenAI opinion on liability & compliance (advisory only) |

---

## Routing Decisions

`auto_settle` · `cat_surge_processing` · `total_loss_routing` · `late_notice_review` ·
`manual_review_route` · `coverage_denial` · `siu_referral` · `senior_review` ·
`adjuster_review`

---

## Setup

```bash
cp .env.example .env
# Optional: set OPENAI_API_KEY for the AI advisory agent
# Optional: set TRACKER_TYPE + Jira/ADO vars to post decisions to a tracker
pip install -r requirements.txt
```

## Run

```bash
# One-command pipeline
python -m backend.pipeline.runner <bundle>            # one claim bundle
python -m backend.pipeline.runner <dir> --all         # every bundle in a folder (+ system metrics)
python -m backend.pipeline.runner <fnol.pdf> --fnol   # single FNOL document
python -m backend.pipeline.runner <bundle> --verify   # byte-for-byte determinism check

# Backend API (Swagger at /docs)
uvicorn backend.main:app --reload

# Frontend console
streamlit run frontend/app.py        # http://localhost:8501
```

---

## Run Artifacts (per claim, in `runs/{claim_id}/`)

| File | Content |
| ---- | ------- |
| `context_packet.json` | FNOL context + evidence index |
| `claim_summary.json` | Extracted fields + confidence + bounding boxes |
| `coverage_result.json` | Coverage validation findings |
| `settlement_calc.json` | Settlement breakdown + line items |
| `fraud_score.json` | Fraud score + indicators + evidence |
| `liability_result.json` | Fault allocation (stretch) |
| `compliance_result.json` | NAIC/state checks (stretch) |
| `anomaly_report.json` | Outlier + surge flags (stretch) |
| `approval_packet.json` | Final routing decision + evidence bundle |
| `exceptions.md` | Exception summary + required actions |
| `audit_log.md` | Step-by-step decision trace (PII-redacted) |
| `metrics.json` | Per-claim metrics (accuracy, exception rate, …) |
| `settlement_payload.json` / `.csv` | CoreLogic-ready posting payload |
| `tracker_ticket.json` | ADO/Jira work-item payload |
| `llm_advisory.json` / `.md` | AI advisory opinion (if generated) |

A batch run (`--all`) also writes `runs/_system_metrics.json` (throughput, mean
extraction accuracy, exception rate, routing distribution).

---

## API Endpoints

`GET /health` · `GET /claims` · `POST /run` · `POST /run-fnol` · `POST /ingest` ·
`POST /tracker/post` · `GET /runs` · `GET /runs/{id}` ·
`GET /runs/{id}/artifacts/{name}` · `GET /policy`

---

## Configuration — `config/policy.yaml`

All operational rules are data-driven (no hardcoding): approval limits, fraud
thresholds and known rings/blacklists, coverage windows and sub-limits, the
catastrophe event, privacy/PII rules, and per-state compliance overrides.
Editing this file changes system behavior without touching code (REQ-058).

---

## Success Criteria

- **Extraction accuracy** — confidence scores + bounding boxes per field.
- **Matching precision** — invoice ↔ PO ↔ GRN three-way matching.
- **Deterministic outputs** — verified byte-for-byte across all scenarios.
- **Auditability** — evidence-first trail with PII redaction.

---

## Testing

```bash
python -m pytest -q
```

The suite covers every agent, the runner, the API, invoice matching, single-FNOL
intake, cross-claim duplicate detection, the stretch agents, OCR graceful
degradation, and tracker posting (mocked). All bundled scenarios route correctly
and are byte-for-byte deterministic.

---

## Project Structure

```text
backend/
  agents/        A–E, H + liability, compliance, anomaly, llm_advisory
  models/        schemas.py  (Pydantic data contracts)
  pipeline/      runner.py   (orchestration, idempotency, determinism, CLI)
  tools/         pdf_extractor, invoice_matcher, fnol_intake, ocr, tracker_poster
  main.py        FastAPI app
frontend/
  app.py         Streamlit claims console
config/
  policy.yaml    dynamic policy pack
datasets/        synthetic claim bundles (12 scenarios)
tests/           automated test suite
runs/            pipeline output artifacts (gitignored)
```

---

## Stretch / Future Work

- Real OCR on scanned photos (the OCR path is implemented with graceful degradation).
- Live posting to a real Jira / Azure DevOps instance (payload + poster implemented).
- Wiring the AI advisory agent deeper into the dashboard workflow.
