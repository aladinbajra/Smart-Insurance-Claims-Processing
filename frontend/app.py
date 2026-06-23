"""
SICPS — Claims Command Center (Streamlit).

An enterprise operational console over the deterministic multi-agent pipeline.
Four consoles: Command Center (triage), Claim Investigation (pipeline + evidence
room), Policy Control (live threshold cockpit), and Analytics. The look is a
self-contained inline-styled design system on a pinned light theme
(.streamlit/config.toml) so every element is always visible and verifiable.
"""

from __future__ import annotations

import html
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline.runner import PipelineRunner  # noqa: E402

CONFIG = str(PROJECT_ROOT / "config" / "policy.yaml")
DATASET = str(PROJECT_ROOT / "datasets" / "SICPS_Dataset (3)" / "SICPS_Dataset")
RUNS = str(PROJECT_ROOT / "runs")

INK = "#0f172a"
MUTED = "#64748b"
LINE = "#e2e8f0"

AGENT_PIPELINE = (
    ("A", "Intake", "context_packet.json", "agent_fnol_intake"),
    ("B", "Extraction", "claim_summary.json", "agent_document_extraction"),
    ("C", "Coverage", "coverage_result.json", "agent_coverage_validation"),
    ("D", "Damage", "settlement_calc.json", "agent_damage_assessment"),
    ("E", "Fraud", "fraud_score.json", "agent_fraud_detection"),
    ("F", "Liability", "liability_result.json", "agent_liability"),
    ("G", "Compliance", "compliance_result.json", "agent_compliance"),
    ("N", "Anomaly", "anomaly_report.json", "agent_anomaly"),
    ("H", "Orchestrator", "approval_packet.json", "agent_exception_triage"),
)

DECISION = {
    "auto_settle": ("Approved — auto-settled", "#16a34a", False),
    "cat_surge_processing": ("Approved — CAT fast-track", "#0891b2", False),
    "total_loss_routing": ("Total loss — valuation", "#be123c", True),
    "late_notice_review": ("Late notice — review", "#d97706", True),
    "manual_review_route": ("Manual review", "#7c3aed", True),
    "adjuster_review": ("Adjuster review", "#ca8a04", True),
    "senior_review": ("Senior review", "#4f46e5", True),
    "coverage_denial": ("Denied — not covered", "#991b1b", True),
    "siu_referral": ("Fraud — SIU referral", "#dc2626", True),
}
SEVERITY = {"critical": "#dc2626", "high": "#ea580c", "medium": "#d97706",
            "low": "#0891b2", "info": "#2563eb"}

st.set_page_config(page_title="SICPS Command Center", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.3rem; max-width: 1480px; }
    section[data-testid="stSidebar"] { border-right: 1px solid #e2e8f0; }
    h1,h2,h3 { color: #0f172a; font-weight: 800; letter-spacing: -0.02em; }
    div[data-testid="stMetric"] { background:#fff; border:1px solid #e2e8f0;
        border-radius:12px; padding:14px 16px; }
    div[data-testid="stMetricValue"] { color:#0f172a; }
    div[data-testid="stMetricLabel"] p { color:#64748b; font-weight:600; }
    div[data-testid="stDataFrame"] { border:1px solid #e2e8f0; border-radius:10px; }
    .stButton button[kind="primary"] { background:#2563eb; border:none; color:#fff; }
    .stButton button { border-radius:8px; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(show_spinner=False)
def _text(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def _yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def money(v: Any) -> str:
    return f"${_f(v):,.0f}"


def clear_cache() -> None:
    _json.clear()
    _text.clear()
    load_claims.clear()


def runner(config: str = CONFIG) -> PipelineRunner:
    return PipelineRunner(config_path=config, runs_dir=RUNS)


@st.cache_data(show_spinner=False)
def load_claims() -> list[dict[str, Any]]:
    root = Path(DATASET)
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for mp in sorted(root.glob("*/manifest.yaml")):
        m = _yaml(str(mp))
        if not m:
            continue
        cid = str(m.get("claim_id", ""))
        rd = Path(RUNS) / cid
        approval = _json(str(rd / "approval_packet.json"))
        summary = _json(str(rd / "claim_summary.json"))
        settlement = _json(str(rd / "settlement_calc.json"))
        fraud = _json(str(rd / "fraud_score.json"))
        coverage = _json(str(rd / "coverage_result.json"))
        routing = approval.get("routing_decision", "")
        exc = approval.get("exceptions", []) or []
        sev = _top_severity(exc)
        out.append({
            "claim_id": cid,
            "claimant": str((m.get("claimant") or {}).get("full_name", "")),
            "scenario": str(m.get("scenario", mp.parent.name)),
            "bundle_path": str(mp.parent),
            "run_dir": str(rd),
            "claim_type": str(m.get("claim_type", "")),
            "priority": str(m.get("priority", "")),
            "processed": bool(approval),
            "routing": routing,
            "attention": DECISION.get(routing, ("", "", False))[2],
            "net": _f(approval.get("net_settlement")),
            "gross": _f(settlement.get("gross_amount")),
            "fraud_score": _f(fraud.get("fraud_score")),
            "risk_level": str(fraud.get("risk_level", "")),
            "coverage_active": coverage.get("coverage_active"),
            "ocr": _f(summary.get("overall_confidence")),
            "top_severity": sev,
            "exceptions": exc,
            "findings": approval.get("all_findings", []) or [],
            "approval": approval, "summary": summary,
            "settlement": settlement, "coverage": coverage, "fraud": fraud,
        })
    return out


def _top_severity(exceptions: list[dict[str, Any]]) -> str:
    order = ["critical", "high", "medium", "low", "info"]
    sevs = [e.get("severity", "") for e in exceptions]
    for s in order:
        if s in sevs:
            return s
    return ""


def sysmetrics() -> dict[str, Any]:
    return _json(str(Path(RUNS) / "_system_metrics.json"))


# ----------------------------------------------------------------------
# Design-system components (inline-styled → always render & visible)
# ----------------------------------------------------------------------

def stat_card(label: str, value: str, sub: str = "", accent: str = "#2563eb") -> None:
    st.markdown(
        f'<div style="background:#fff;border:1px solid {LINE};border-left:4px solid {accent};'
        f'border-radius:12px;padding:15px 18px;box-shadow:0 1px 2px rgba(15,23,42,.05);">'
        f'<div style="color:{MUTED};font-size:.7rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.06em;">{html.escape(label)}</div>'
        f'<div style="color:{INK};font-size:1.7rem;font-weight:800;line-height:1.15;margin-top:3px;">{value}</div>'
        f'<div style="color:#94a3b8;font-size:.76rem;margin-top:2px;">{html.escape(sub)}</div></div>',
        unsafe_allow_html=True,
    )


def pill(text: str, color: str) -> str:
    return (f'<span style="display:inline-block;padding:3px 12px;border-radius:999px;'
            f'background:{color};color:#fff;font-size:.76rem;font-weight:700;">{html.escape(text)}</span>')


def verdict_banner(routing: str, net: float) -> None:
    label, color, _ = DECISION.get(routing, ("Not processed", MUTED, False))
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{color}14,transparent);'
        f'border:1px solid {color}55;border-left:6px solid {color};border-radius:12px;'
        f'padding:16px 20px;display:flex;justify-content:space-between;align-items:center;">'
        f'<div><div style="color:{MUTED};font-size:.72rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.06em;">Routing decision</div>'
        f'<div style="color:{color};font-size:1.5rem;font-weight:800;">{html.escape(label)}</div></div>'
        f'<div style="text-align:right;"><div style="color:{MUTED};font-size:.72rem;font-weight:700;'
        f'text-transform:uppercase;">Net settlement</div>'
        f'<div style="color:{INK};font-size:1.5rem;font-weight:800;">{money(net)}</div></div></div>',
        unsafe_allow_html=True,
    )


def pipeline_timeline(claim: dict[str, Any]) -> None:
    run_dir = Path(claim["run_dir"])
    findings = claim.get("findings", [])
    cells = []
    for i, (letter, name, artifact, agent_key) in enumerate(AGENT_PIPELINE):
        present = (run_dir / artifact).is_file()
        alert = any(f.get("agent") == agent_key and f.get("severity") in ("critical", "high")
                    for f in findings)
        color = "#dc2626" if alert else ("#16a34a" if present else "#cbd5e1")
        glyph = "!" if alert else ("✓" if present else "·")
        connector = (f'<div style="flex:1;height:2px;background:'
                     f'{"#16a34a" if present else "#e2e8f0"};margin-top:21px;"></div>') if i else ""
        cells.append(connector)
        cells.append(
            f'<div style="text-align:center;min-width:78px;">'
            f'<div style="width:44px;height:44px;line-height:44px;border-radius:12px;margin:0 auto;'
            f'background:{color};color:#fff;font-weight:800;font-size:1.05rem;'
            f'box-shadow:0 2px 6px {color}40;">{letter}<span style="font-size:.7rem;"> {glyph}</span></div>'
            f'<div style="font-size:.72rem;font-weight:600;color:#334155;margin-top:6px;">{name}</div></div>'
        )
    st.markdown(
        '<div style="display:flex;align-items:flex-start;gap:4px;overflow-x:auto;padding:8px 2px;">'
        + "".join(cells) + "</div>",
        unsafe_allow_html=True,
    )


def evidence_room(claim: dict[str, Any]) -> None:
    fields = claim["summary"].get("extracted_fields", []) or []
    if not fields:
        st.caption("No extracted fields available.")
        return
    rows = []
    for fld in sorted(fields, key=lambda x: -_f(x.get("confidence"))):
        name = html.escape(str(fld.get("field_name", "")))
        val = html.escape(str(fld.get("value", ""))[:60])
        conf = _f(fld.get("confidence"))
        c_color = "#16a34a" if conf >= 0.7 else "#d97706" if conf >= 0.6 else "#dc2626"
        bbox = fld.get("bbox") or {}
        src = html.escape(str(fld.get("source_doc", "")).split(" ")[0])
        loc = (f'{src}:p{bbox.get("page")}·({bbox.get("x1")},{bbox.get("y1")})'
               if bbox else src or "—")
        rows.append(
            f'<div style="display:grid;grid-template-columns:1.3fr 1.6fr 1.3fr 1.8fr;gap:10px;'
            f'align-items:center;padding:8px 10px;border-bottom:1px solid #f1f5f9;">'
            f'<div style="font-weight:600;color:#334155;font-size:.82rem;">{name}</div>'
            f'<div style="color:#0f172a;font-size:.82rem;">{val or "—"}</div>'
            f'<div><div style="background:#eef2f7;border-radius:999px;height:8px;width:100%;">'
            f'<div style="background:{c_color};height:8px;border-radius:999px;width:{conf*100:.0f}%;"></div></div>'
            f'<div style="font-size:.68rem;color:{c_color};font-weight:700;margin-top:2px;">{conf*100:.0f}%</div></div>'
            f'<div style="color:#94a3b8;font-size:.72rem;font-family:monospace;">{html.escape(loc)}</div></div>'
        )
    st.markdown(
        '<div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;">'
        '<div style="display:grid;grid-template-columns:1.3fr 1.6fr 1.3fr 1.8fr;gap:10px;'
        'padding:9px 10px;background:#f8fafc;border-bottom:1px solid #e2e8f0;'
        'font-size:.68rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.04em;">'
        '<div>Field</div><div>Value</div><div>Confidence</div><div>Source · bbox</div></div>'
        + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------

def _layout(fig: go.Figure, h: int = 300, title: str = "") -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40 if title else 12, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", color="#334155", size=12),
                      title=dict(text=title, font=dict(color=INK, size=15), x=0.01),
                      showlegend=False)
    fig.update_xaxes(gridcolor="rgba(148,163,184,.16)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.16)")
    return fig


def routing_donut(claims: list[dict[str, Any]]) -> go.Figure:
    counts: dict[str, int] = {}
    for c in claims:
        if c["processed"]:
            counts[c["routing"]] = counts.get(c["routing"], 0) + 1
    keys = sorted(counts)
    fig = go.Figure(go.Pie(
        labels=[DECISION.get(k, (k, "", False))[0] for k in keys],
        values=[counts[k] for k in keys], hole=0.62,
        marker=dict(colors=[DECISION.get(k, ("", "#64748b", False))[1] for k in keys]),
    ))
    return _layout(fig, 320, "Decision mix")


def fraud_gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        gauge={"axis": {"range": [0, 1]},
               "bar": {"color": "#dc2626" if score >= 0.75 else "#d97706" if score >= 0.4 else "#16a34a"},
               "steps": [{"range": [0, 0.4], "color": "#dcfce7"},
                         {"range": [0.4, 0.75], "color": "#fef3c7"},
                         {"range": [0.75, 1], "color": "#fee2e2"}]},
    ))
    return _layout(fig, 240, "Fraud score")


def severity_bar(claims: list[dict[str, Any]]) -> go.Figure:
    counts: dict[str, int] = {}
    for c in claims:
        for f in c.get("findings", []):
            s = f.get("severity", "info")
            counts[s] = counts.get(s, 0) + 1
    order = [s for s in ("critical", "high", "medium", "low", "info") if s in counts]
    fig = go.Figure(go.Bar(x=order, y=[counts[s] for s in order],
                           marker_color=[SEVERITY[s] for s in order]))
    return _layout(fig, 300, "Findings by severity")


def settlement_bar(claims: list[dict[str, Any]]) -> go.Figure:
    run = [c for c in claims if c["processed"]]
    fig = go.Figure(go.Bar(x=[c["claim_id"] for c in run], y=[c["net"] for c in run],
                           marker_color="#2563eb"))
    return _layout(fig, 320, "Net settlement by claim")


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------

def command_center(claims: list[dict[str, Any]]) -> None:
    st.markdown("## Command Center")
    st.caption("Live portfolio status and triage — what needs a human right now.")
    proc = [c for c in claims if c["processed"]]
    approved = sum(1 for c in proc if c["routing"] in ("auto_settle", "cat_surge_processing"))
    siu = sum(1 for c in proc if c["routing"] == "siu_referral")
    denied = sum(1 for c in proc if c["routing"] == "coverage_denial")
    review = sum(1 for c in proc if c["attention"] and c["routing"] not in ("siu_referral", "coverage_denial"))
    payout = sum(c["net"] for c in proc)
    sm = sysmetrics()

    cols = st.columns(6)
    with cols[0]: stat_card("Claims", str(len(claims)), f"{len(proc)} processed", "#2563eb")
    with cols[1]: stat_card("Approved", str(approved), "straight-through", "#16a34a")
    with cols[2]: stat_card("Needs review", str(review), "human queue", "#d97706")
    with cols[3]: stat_card("Fraud / SIU", str(siu), "investigation", "#dc2626")
    with cols[4]: stat_card("Denied", str(denied), "no payment", "#991b1b")
    with cols[5]: stat_card("Total payout", money(payout), "net settled", "#0891b2")

    if sm:
        c2 = st.columns(4)
        with c2[0]: stat_card("Throughput", f"{sm.get('throughput_claims_per_sec', 0)}/s", "batch", "#4f46e5")
        with c2[1]: stat_card("Extraction accuracy", f"{_f(sm.get('mean_extraction_accuracy'))*100:.0f}%", "mean", "#0891b2")
        with c2[2]: stat_card("Exception rate", f"{_f(sm.get('exception_rate'))*100:.0f}%", "of findings", "#d97706")
        with c2[3]: stat_card("Cache hits", str(sm.get("cache_hits", 0)), "idempotent", "#16a34a")

    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(routing_donut(claims), use_container_width=True)
    with right:
        st.plotly_chart(severity_bar(claims), use_container_width=True)

    attention = sorted([c for c in proc if c["attention"]],
                       key=lambda c: ["critical", "high", "medium", "low", "info", ""].index(c["top_severity"]))
    st.markdown("### Triage queue — needs attention")
    if not attention:
        st.success("Queue clear — no claims require manual action.")
        return
    for c in attention:
        color = DECISION.get(c["routing"], ("", MUTED, True))[1]
        sev = c["top_severity"] or "info"
        st.markdown(
            f'<div style="background:#fff;border:1px solid {LINE};border-left:5px solid {color};'
            f'border-radius:10px;padding:12px 16px;margin-bottom:8px;display:flex;'
            f'justify-content:space-between;align-items:center;">'
            f'<div><span style="font-weight:700;color:{INK};">{c["claim_id"]}</span> '
            f'<span style="color:{MUTED};">· {html.escape(c["claimant"] or c["scenario"])}</span><br>'
            f'<span style="color:{MUTED};font-size:.8rem;">{DECISION.get(c["routing"],("",))[0]}</span></div>'
            f'<div style="display:flex;gap:10px;align-items:center;">{pill(sev, SEVERITY.get(sev, MUTED))}'
            f'<span style="font-weight:700;color:{INK};">{money(c["net"])}</span></div></div>',
            unsafe_allow_html=True,
        )


def investigation(claims: list[dict[str, Any]]) -> None:
    st.markdown("## Claim Investigation")
    opts = {f'{c["claim_id"]} — {c["claimant"] or c["scenario"]}': c for c in claims}
    c = opts[st.selectbox("Select claim", list(opts))]

    a = st.columns(4)
    if a[0].button("▶ Process" if not c["processed"] else "⟳ Re-process",
                   type="primary", use_container_width=True, key="proc"):
        with st.spinner("Running the deterministic pipeline…"):
            runner().run(c["bundle_path"], force=True)
            clear_cache()
        st.rerun()
    if a[1].button("✓ Verify Determinism", use_container_width=True, key="ver"):
        with st.spinner("Running twice, comparing byte-for-byte…"):
            ok = runner().verify_determinism(c["bundle_path"])
        (st.success if ok else st.error)(f"Determinism: {'PASS' if ok else 'FAIL'}")
    if a[2].button("🤖 AI Advisory", use_container_width=True, key="adv"):
        from dotenv import load_dotenv
        from backend.agents.agent_llm_advisory import AgentLLMAdvisory
        load_dotenv()
        with st.spinner("Asking the AI advisor…"):
            AgentLLMAdvisory().process(c["run_dir"]); clear_cache()
        st.rerun()
    if a[3].button("📤 Post to Tracker", use_container_width=True, key="trk"):
        from dotenv import load_dotenv
        from backend.tools.tracker_poster import post_ticket_file
        load_dotenv()
        r = post_ticket_file(c["run_dir"])
        (st.success if r.get("posted") else st.info)(
            r.get("action", "posted") if r.get("posted") else (r.get("reason") or "dry-run"))

    if not c["processed"]:
        st.info("This claim has not been processed. Click **Process**.")
        return

    verdict_banner(c["routing"], c["net"])
    st.markdown("#### Deterministic agent pipeline")
    pipeline_timeline(c)

    tabs = st.tabs(["🔬 Evidence Room", "⚠ Exceptions", "🛡 Coverage & Fraud", "📜 Audit Trail", "🤖 AI Advisory"])
    with tabs[0]:
        st.caption("Every field the agents extracted — value, confidence, and source location in the PDF.")
        evidence_room(c)
    with tabs[1]:
        exc = c["exceptions"]
        if exc:
            st.dataframe([{"Issue": e.get("category", "").replace("_", " "),
                           "Severity": e.get("severity"),
                           "Action": e.get("recommended_action"),
                           "Owner": e.get("assigned_to") or "—"} for e in exc],
                         use_container_width=True, hide_index=True)
        else:
            st.success("No exceptions — straight-through.")
    with tabs[2]:
        m = st.columns([1, 1, 1.4])
        cov = c["coverage"]
        with m[0]:
            stat_card("Coverage", "Active" if cov.get("coverage_active") else "Not active",
                      cov.get("denial_reason") or "", "#16a34a" if cov.get("coverage_active") else "#dc2626")
            stat_card("Gross", money(c["gross"]), "before deductible", "#2563eb")
        with m[1]:
            stat_card("Deductible", money(cov.get("deductible")), "policy", "#64748b")
            stat_card("Net payout", money(c["net"]), "final", "#0891b2")
        with m[2]:
            st.plotly_chart(fraud_gauge(c["fraud_score"]), use_container_width=True)
    with tabs[3]:
        st.markdown(_text(str(Path(c["run_dir"]) / "audit_log.md")) or "No audit log.")
    with tabs[4]:
        adv = _json(str(Path(c["run_dir"]) / "llm_advisory.json"))
        if adv.get("status") == "ok":
            lia, comp = adv.get("liability") or {}, adv.get("compliance") or {}
            st.markdown(f"**Liability** — at fault: {lia.get('at_fault_party','—')} "
                        f"(insured {lia.get('insured_liability_pct',0)}% / claimant {lia.get('claimant_liability_pct',0)}%)")
            st.markdown(f"**Compliance** — {comp.get('framework','—')}: "
                        f"{'compliant' if comp.get('compliant') else 'review needed'}")
            if adv.get("narrative"):
                st.info(adv["narrative"])
        else:
            st.caption("No AI advisory yet — click **AI Advisory** above (needs OPENAI_API_KEY).")


def policy_control(claims: list[dict[str, Any]]) -> None:
    st.markdown("## Policy Control")
    st.caption("Adjust the live decision thresholds and simulate the impact on a claim — "
               "no code, no risk (simulation uses a temporary policy file).")
    pol = _yaml(CONFIG)
    appr = pol.get("approval", {}) or {}
    fr = pol.get("fraud", {}) or {}

    left, right = st.columns(2)
    with left:
        auto = st.slider("Auto-settle limit ($)", 0, 20000, int(_f(appr.get("auto_settle_max_amount", 5000))), 250)
        senior = st.slider("Senior-review threshold ($)", 10000, 100000, int(_f(appr.get("senior_review_threshold", 50000))), 1000)
        tl = st.slider("Total-loss ratio", 0.0, 1.5, _f(appr.get("total_loss_ratio_threshold", 0.75)), 0.05)
    with right:
        high = st.slider("Fraud → SIU threshold", 0.0, 1.0, _f(fr.get("high_risk_threshold", 0.75)), 0.05)
        med = st.slider("Fraud → review threshold", 0.0, 1.0, _f(fr.get("medium_risk_threshold", 0.40)), 0.05)
        prior = st.slider("Prior-claims → SIU", 1, 8, int(_f(fr.get("prior_claims_siu_trigger", 4))), 1)

    opts = {f'{c["claim_id"]} — {c["scenario"]}': c for c in claims}
    target = opts[st.selectbox("Simulate on claim", list(opts))]

    if st.button("⚡ Simulate impact", type="primary"):
        sim = _yaml(CONFIG)
        sim.setdefault("approval", {}).update({
            "auto_settle_max_amount": float(auto),
            "senior_review_threshold": float(senior),
            "total_loss_ratio_threshold": float(tl)})
        sim.setdefault("fraud", {}).update({
            "high_risk_threshold": float(high),
            "medium_risk_threshold": float(med),
            "prior_claims_siu_trigger": int(prior)})
        sandbox = Path(tempfile.mkdtemp())
        tmp = sandbox / "policy_sim.yaml"
        tmp.write_text(yaml.safe_dump(sim, sort_keys=False), encoding="utf-8")
        before = target["routing"] or "—"
        with st.spinner("Re-running the claim under the new thresholds…"):
            # Isolated sandbox runs dir — simulation never touches the canonical
            # runs/ artifacts, so the live decision state is preserved.
            res = PipelineRunner(config_path=str(tmp), runs_dir=str(sandbox)).run(
                target["bundle_path"], force=True)
        bcol, acol = st.columns(2)
        with bcol:
            stat_card("Decision before", before.replace("_", " "), "live policy", "#64748b")
        with acol:
            changed = res.routing_decision != before
            stat_card("Decision after", res.routing_decision.replace("_", " "),
                      "changed ✓" if changed else "unchanged",
                      "#16a34a" if changed else "#64748b")
        st.caption("This proves configurability: the same claim, different thresholds, a different decision — "
                   "with zero code changes. The live policy.yaml was not modified.")


def analytics(claims: list[dict[str, Any]]) -> None:
    st.markdown("## Analytics")
    st.caption("Portfolio-level settlement, decision, and risk analytics.")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(settlement_bar(claims), use_container_width=True)
        st.plotly_chart(severity_bar(claims), use_container_width=True)
    with right:
        st.plotly_chart(routing_donut(claims), use_container_width=True)
        proc = [c for c in claims if c["processed"]]
        avg = sum(c["fraud_score"] for c in proc) / len(proc) if proc else 0.0
        st.plotly_chart(fraud_gauge(avg), use_container_width=True)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    with st.sidebar:
        st.markdown("## ⬡ SICPS")
        st.caption("Claims Command Center")
        page = st.radio("Navigate", ("Command Center", "Claim Investigation",
                                      "Policy Control", "Analytics"),
                        label_visibility="collapsed")
        st.divider()
        force = st.toggle("Always re-process", value=False)
        if st.button("⏵ Process all claims", use_container_width=True):
            with st.spinner("Processing the full portfolio…"):
                runner().run_all(DATASET, force=force)
                clear_cache()
            st.rerun()
        if st.button("↻ Refresh", use_container_width=True):
            clear_cache()
            st.rerun()
        st.divider()
        st.caption("Deterministic · file-based · auditable")

    claims = load_claims()
    if not claims:
        st.markdown("## No claims found")
        st.caption(f"Looked under: {DATASET}")
        return

    if page == "Command Center":
        command_center(claims)
    elif page == "Claim Investigation":
        investigation(claims)
    elif page == "Policy Control":
        policy_control(claims)
    else:
        analytics(claims)


if __name__ == "__main__":
    main()
