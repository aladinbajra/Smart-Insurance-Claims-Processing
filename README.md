# Smart Insurance Claims Processing (SICPS)

Multi-agent AI pipeline for automated insurance claims validation and settlement.
Built with Python, FastAPI, Anthropic Claude, and Streamlit.

---

## Pipeline — 6 Agents

| Agent | File | Responsibility |
| ----- | ---- | -------------- |
| A | `agent_fnol_intake.py` | FNOL Intake + Classification (Gatekeeper) |
| B | `agent_document_extraction.py` | Document Extraction — structured fields + confidence scores |
| C | `agent_coverage_validation.py` | Coverage Validation — exclusions, deductibles, waiting periods |
| D | `agent_damage_assessment.py` | Damage Assessment — gross claim calculation |
| E | `agent_fraud_detection.py` | Fraud Detection — scoring, SIU referral |
| H | `agent_exception_triage.py` | Exception Triage + Adjuster Routing + Settlement Payload |

---

## Project Structure

```text
├── backend/
│   ├── agents/
│   │   ├── agent_fnol_intake.py
│   │   ├── agent_document_extraction.py
│   │   ├── agent_coverage_validation.py
│   │   ├── agent_damage_assessment.py
│   │   ├── agent_fraud_detection.py
│   │   └── agent_exception_triage.py
│   ├── models/
│   │   └── schemas.py
│   ├── pipeline/
│   │   └── runner.py
│   ├── tools/
│   │   └── pdf_extractor.py
│   └── main.py
├── frontend/
│   └── app.py
├── config/
│   └── policy.yaml
├── tests/
│   └── fixtures/
│       ├── scenario_01_clean_auto/
│       ├── scenario_02_total_loss/
│       ├── scenario_03_siu_referral/
│       ├── scenario_04_coverage_denial/
│       ├── scenario_05_late_notice/
│       ├── scenario_06_cat_event/
│       ├── scenario_07_fraud_ring/
│       ├── scenario_08_low_ocr/
│       └── scenario_09_homeowners/
├── claims/              # Input claim bundles (gitignored)
├── runs/                # Pipeline output artifacts (gitignored)
├── .env.example
└── requirements.txt
```

---

## Setup

```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
pip install -r requirements.txt
```

## Run

```bash
# Backend API
uvicorn backend.main:app --reload

# Frontend dashboard
streamlit run frontend/app.py
```

---

## Output Artifacts (per run)

Each pipeline run produces the following in `runs/{claim_id}/`:

| File | Content |
| ---- | ------- |
| `context_packet.json` | FNOL context + evidence index |
| `claim_summary.json` | Extracted fields + confidence scores |
| `coverage_result.json` | Coverage validation findings |
| `settlement_calc.json` | Damage totals + line items |
| `fraud_score.json` | Fraud risk score + indicators |
| `approval_packet.json` | Routing decision + adjuster info |
| `exceptions.md` | Exception summary + next actions |
| `audit_log.md` | Step-by-step trace of all decisions |
| `metrics.json` | Throughput + accuracy rates |
