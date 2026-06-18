# Bug Report — Agent C (Coverage Validation) & Agent D (Damage Assessment)

**Tested:** 2026-06-15
**Method:** Ran Agent A → loaded `context_packet.json` → fed it to Agent C and Agent D across all 12 dataset scenarios, comparing actual output against the manifest ground-truth (`financials`, `flags`, `expected_agent_signals`).
**Scope:** No code was modified. This document only reports findings.

---

## Summary

| ID | Component | Severity | One-line |
|----|-----------|----------|----------|
| BUG-X1 | C + D | **High** | C and D read `context_packet` (manifest values), not Agent B's `claim_summary.json` — extraction + confidence work is bypassed |
| BUG-C1 | Agent C | **High** | Agent C never writes `coverage_result.json` to disk — breaks the file-based pipeline contract |
| BUG-C2 | Agent C | **High** | Coverage decision reads pre-computed answer flags instead of deriving it — fails on any real claim without pre-set flags |
| BUG-D1 | Agent D | **High** | `total_loss_ratio = gross/market` (includes medical) instead of `repair/market` — scenario_07 wrongly flagged total loss |
| BUG-D2 | Agent D | **High** | Total-loss threshold hardcoded `0.75` instead of reading `policy.yaml` — breaks REQ-058 configurability demo |
| BUG-C3 | Agent C | Medium | No state overrides applied from `policy.yaml`; IN override (45) disagrees with manifest (30) |
| BUG-C4 | Agent C | Medium | Policy exclusions never evaluated — `exclusions_triggered` always empty (REQ-017 unmet) |
| BUG-C5 | Agent C | Medium | `denial_reason` is free text, not the machine codes Agent H expects |
| BUG-D3 | Agent D | Medium | `repair_variance_pct` is meaningless (computes repair/market) — spurious findings on 5 scenarios |
| BUG-D4 | Agent D | Medium | `medical_variance_pct` is meaningless (computes medical/gross) — spurious fin/ding on scenario_12 |
| BUG-D5 | Agent D | Medium | `net_settlement` returns 0.0 for low-OCR/missing-docs claims; should be null/unresolvable |
| BUG-C6 | Agent C | Low | `waiting_period_active` hardcoded False; `pre_existing_lookback_days` never used |
| BUG-C7 | Agent C | Low | Late-notice rule still fires after a policy is already expired/denied (double findings) |
| BUG-D6 | Agent D | Low | `gross_amount` excludes `injury_claim`; differs from manifest gross (partly a dataset inconsistency) |
| BUG-CD7 | C + D | Low | `datetime.utcnow()` is deprecated in Python 3.12 |

---

## CROSS-CUTTING

### BUG-X1 — Agents C and D consume the manifest, not Agent B's extraction output
**Severity:** High
**Component:** Agent C, Agent D (architecture)

**Description:**
Both agents call `process(context_packet)` and read values out of `context_packet["financials"] / ["policy"] / ["incident"] / ["flags"]`. The context packet is Agent A's output, populated **directly from `manifest.yaml`** (ground truth). They never read `claim_summary.json` (Agent B's output).

**Impact:**
- Agent B's entire job — PDF extraction, per-field confidence, low-OCR detection, missing-field handling — is invisible to C and D. The pipeline effectively bypasses Agent B for all downstream math.
- Quality signals (`requires_manual_review`, low confidence, missing required document) never reach the financial/coverage logic. This is the root cause of BUG-D5.
- On a real claim, the manifest's pre-computed `financials`/`flags` would not exist, so C and D would have nothing to read.

**Expected:** Per the file-based coordination contract (REQ-038–039), Agent C reads `claim_summary.json` + `policy.yaml`; Agent D reads `claim_summary.json` + `coverage_result.json`.

---

## AGENT C — Coverage Validation

### BUG-C1 — Agent C never persists `coverage_result.json`
**Severity:** High
**Steps to reproduce:** Call `AgentCoverageValidation().process(context_packet)` for any scenario.
**Expected:** `runs/{claim_id}/coverage_result.json` is written (Agent D and Agent H read it from disk).
**Actual:** `process()` returns a `CoverageResult` object only. The class has no `runs_dir`, no `_write_json`, and writes nothing. Agents A, B, and D all write their JSON; C does not.
**Impact:** The runner/Agent H expect `coverage_result.json` on disk. As-is, that file never exists and the pipeline chain is broken at C.

### BUG-C2 — Coverage decision reads pre-computed answers instead of deriving them
**Severity:** High
**Evidence:**
- Expired check: `if policy["policy_expired"]:` (reads the manifest's answer flag).
- Denial check: `elif flags["coverage_denied"]:` (reads the manifest's answer flag).
**Expected:** Derive expiry from `incident.date` vs `policy.expiration_date`; derive denial from `incident.date` vs `policy.effective_date` (pre-inception) and from exclusion matching.
**Actual:** Agent C trusts `flags.coverage_denied` / `policy.policy_expired`, which are pre-set in the synthetic dataset. It works only because the dataset hands it the answer.
**Impact:** On a real claim (no pre-set flags), Agent C cannot detect an expired policy or a pre-inception incident. It is reading the test answer, not validating coverage.

### BUG-C3 — State overrides from `policy.yaml` are not applied
**Severity:** Medium
**Evidence:** Late-notice rule uses `policy["late_notice_days"]` (from the manifest). `policy.yaml → compliance.state_overrides` (IL:30, IN:45, OH:30) is never read.
**Concrete discrepancy:** scenario_02 is state **IN**; the manifest sets `late_notice_days: 30`, but `policy.yaml` defines IN override = **45**. The two disagree and Agent C silently uses the manifest's 30.
**Impact:** REQ-058 configurability gap — editing the state override in `policy.yaml` has no effect on Agent C.

### BUG-C4 — Policy exclusions are never evaluated
**Severity:** Medium
**Evidence:** `exclusions_triggered = []` is initialized and never populated.
**Expected (REQ-017):** Compare the incident against `policy.exclusions` (e.g. "Damage prior to policy inception", "Pre-existing damage"). scenario_04 should trigger a pre-inception exclusion.
**Actual:** `exclusions_triggered` is always empty for all 12 scenarios.

### BUG-C5 — `denial_reason` is free text, not a machine code
**Severity:** Medium
**Evidence:** Agent C emits `"Policy expired"` and `"Coverage denied"`.
**Expected (per `expected_agent_signals`):** `policy_expired` (sc10), `incident_predates_policy_inception` (sc04), `late_notice_violation` (sc05).
**Impact:** Agent H routing logic keyed on reason codes will not match these free-text strings. For scenario_05, Agent C sets no `denial_reason` at all even though the expected signal is `late_notice_violation`.

### BUG-C6 — Waiting period / pre-existing lookback unimplemented
**Severity:** Low
**Evidence:** `waiting_period_active` is hardcoded `False`; `policy.yaml → coverage.pre_existing_lookback_days` is never read.

### BUG-C7 — Late-notice finding fires even after denial
**Severity:** Low
**Evidence:** scenario_04 returns BOTH `C-002-COVERAGE_DENIED` and `C-003-LATE_NOTICE` because the late-notice check is not guarded by the earlier deny branches.
**Impact:** Duplicate / overlapping findings reach Agent H for an already-denied claim.

---

## AGENT D — Damage Assessment

### BUG-D1 — `total_loss_ratio` includes medical costs (should be repair ÷ market)
**Severity:** High
**Evidence:** `total_loss_ratio = gross_amount / market_value`, where `gross = repair + medical + replacement`.
**Expected (spec + manifest):** `total_loss_ratio = repair_estimate / market_value`.
**Concrete failure — scenario_07 (fraud ring):**
- Agent D: ratio `0.95`, `total_loss_flag = True`
- Manifest: ratio `0.6921`, `total_loss_flag = False`
- The $9,800 medical claim inflates the ratio past 0.75 and wrongly marks the vehicle a total loss.
**Also affects** the ratio value (not the flag) on scenario_02 (1.2125 vs 1.0813) and scenario_12 (0.6947 vs 0.4789).
**Impact:** Wrong `total_loss_flag` in `settlement_calc.json`; could mis-route any claim with significant medical costs to total-loss handling.

### BUG-D2 — Total-loss threshold hardcoded `0.75`
**Severity:** High
**Evidence:** `total_loss_flag = total_loss_ratio >= 0.75` (literal). `self.policy_pack.approval.total_loss_ratio_threshold` is loaded but never used here.
**Impact:** Directly breaks the REQ-058 demo: an evaluator who changes `total_loss_ratio_threshold` in `policy.yaml` sees no change in Agent D behavior. The audit flagged this exact risk ("If hardcoded in Python, this fails").

### BUG-D3 — `repair_variance_pct` does not measure variance
**Severity:** Medium
**Evidence:** `_calculate_repair_variance` returns `repair_estimate / market_value` (i.e. the total-loss ratio again), then compares it against `repair_cost_variance_pct` (0.25).
**Expected:** Variance = submitted estimate vs **expected** repair cost (manifest `invoice_matching.variance_pct`), tolerance ±25%.
**Actual / spurious findings:** `D-003-REPAIR_VARIANCE_EXCEEDED` fires whenever repair > 25% of market value — scenarios 02, 03, 06, 07, 12. Example: scenario_03 has real variance 0.10 (within tolerance) but is flagged because 5220/16000 = 0.326 > 0.25.

### BUG-D4 — `medical_variance_pct` does not measure variance
**Severity:** Medium
**Evidence:** `_calculate_medical_variance` returns `medical_total / gross_amount`, compared against `medical_cost_variance_pct` (0.30).
**Actual:** scenario_12 (medical 8200 / gross 26400 = 0.31) trips `D-004-MEDICAL_VARIANCE_EXCEEDED`, which is just a proportion of the bill, not a deviation from an expected medical cost.

### BUG-D5 — `net_settlement` is 0.0 for unresolvable claims (should be null)
**Severity:** Medium
**Evidence:**
- scenario_08 (low OCR): Agent D net `0.0`; manifest `expected_agent_signals.net_settlement: null, reason: low_ocr_confidence`.
- scenario_11 (missing docs): Agent D net `0.0`; manifest expects `null, reason: missing_documents`.
**Root cause:** Tied to BUG-X1 — Agent D reads manifest financials and has no visibility into Agent B's `requires_manual_review`, so it computes a number where it should report "unresolvable".

### BUG-D6 — `gross_amount` excludes `injury_claim`
**Severity:** Low
**Evidence:** Agent D `gross = repair + medical + replacement`. scenario_02 → 38800; manifest `financials.gross_amount` → 43000 (adds injury 4200). Note the dataset is itself inconsistent (sc02 `expected_agent_signals.gross_amount` = 33300, a third value), so this is partly a data-quality issue to confirm with the dataset owner.

### BUG-CD7 — `datetime.utcnow()` deprecated
**Severity:** Low
**Evidence:** Both agents call `datetime.utcnow().isoformat()`. Deprecated in Python 3.12; use `datetime.now(datetime.UTC)`.

---

## Recommended fix priority (for the demo)

1. **BUG-C1** (write `coverage_result.json`) and **BUG-X1** (read `claim_summary.json`) — without these the pipeline cannot chain A→B→C→D→E→H.
2. **BUG-D2** (configurable total-loss threshold) — required for the REQ-058 live demo.
3. **BUG-D1** (repair/market ratio) — fixes the scenario_07 total-loss mis-flag.
4. **BUG-C2 / C5** — needed so Agent H routing keys on real, derived reason codes.
5. **BUG-D3 / D4 / C3 / C4** — remove spurious findings and meet REQ-017 / REQ-058 fully.
6. Low-severity items as time permits.
