from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline.runner import PipelineRunner  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "policy.yaml"
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "SICPS_Dataset (3)" / "SICPS_Dataset"
DEFAULT_RUNS = PROJECT_ROOT / "runs"

JSON_ARTIFACTS = (
    "context_packet.json",
    "evidence_index.json",
    "claim_summary.json",
    "coverage_result.json",
    "settlement_calc.json",
    "fraud_score.json",
    "approval_packet.json",
    "metrics.json",
    "settlement_payload.json",
)
TEXT_ARTIFACTS = (
    "exceptions.md",
    "audit_log.md",
    "audit_log_agent_b.md",
    "settlement_audit.md",
)
CSV_ARTIFACTS = (
    "settlement_payload.csv",
)
ALL_ARTIFACTS = JSON_ARTIFACTS + TEXT_ARTIFACTS + CSV_ARTIFACTS

PAGES = (
    "Executive Dashboard",
    "Claims Processing",
    "Coverage Intelligence",
    "Settlement Analytics",
    "Fraud Intelligence Center",
    "Audit & Compliance",
    "System Monitoring",
)

AGENT_PIPELINE = (
    ("A", "FNOL Intake", "context_packet.json", "agent_fnol_intake"),
    ("B", "Document Extraction", "claim_summary.json", "agent_document_extraction"),
    ("C", "Coverage Validation", "coverage_result.json", "agent_coverage_validation"),
    ("D", "Damage Assessment", "settlement_calc.json", "agent_damage_assessment"),
    ("E", "Fraud Detection", "fraud_score.json", "agent_fraud_detection"),
    ("F", "Exception Triage", "approval_packet.json", "agent_exception_triage"),
    ("G", "Decision Orchestration", "approval_packet.json", "agent_exception_triage"),
    ("H", "Audit & Reporting", "audit_log.md", "agent_exception_triage"),
)

SEVERITY_COLORS = {
    "critical": "#dc2626",
    "high": "#ef4444",
    "medium": "#f59e0b",
    "low": "#14b8a6",
    "info": "#2563eb",
}

ROUTING_COLORS = {
    "auto_settle": "#059669",
    "adjuster_review": "#f59e0b",
    "siu_referral": "#dc2626",
    "coverage_denial": "#991b1b",
    "late_notice_review": "#d97706",
    "total_loss_routing": "#be123c",
    "cat_surge_processing": "#0284c7",
    "manual_review_route": "#7c3aed",
    "senior_review": "#4f46e5",
    "not_run": "#64748b",
}

# Routing buckets for KPI rollups — kept in sync with the 9 RoutingDecision
# values so summary counts cover every category (auto-settle + CAT surge are
# straight-through; total loss / late notice / manual / senior / adjuster all
# need a human; coverage denial is terminal).
AUTO_ROUTINGS = {"auto_settle", "cat_surge_processing"}
REVIEW_ROUTINGS = {
    "adjuster_review",
    "manual_review_route",
    "senior_review",
    "total_loss_routing",
    "late_notice_review",
}
DENIAL_ROUTINGS = {"coverage_denial"}


st.set_page_config(
    page_title="SICPS Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #0f172a;
        --muted: #64748b;
        --line: rgba(148, 163, 184, 0.22);
        --blue: #2563eb;
        --cyan: #06b6d4;
        --teal: #14b8a6;
        --green: #059669;
        --orange: #f59e0b;
        --red: #dc2626;
        --panel: rgba(255, 255, 255, 0.82);
        --shadow: 0 20px 60px rgba(15, 23, 42, 0.08);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: #0f172a !important;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(14, 165, 233, 0.12), transparent 30rem),
            radial-gradient(circle at top right, rgba(20, 184, 166, 0.12), transparent 26rem),
            linear-gradient(180deg, #f8fbff 0%, #eef7fb 42%, #f9fbff 100%);
    }

    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1480px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,.92), rgba(239,248,255,.86));
        border-right: 1px solid var(--line);
    }

    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label {
        color: #0f172a !important;
        opacity: 1 !important;
    }

    h1, h2, h3, h4, h5, h6,
    p, span, label,
    div[data-testid="stMarkdownContainer"],
    div[data-testid="stMarkdownContainer"] *,
    div[data-testid="stText"],
    div[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] *,
    .stRadio label,
    .stSelectbox label,
    .stTextInput label,
    .stMultiSelect label,
    .stCheckbox label {
        color: #0f172a !important;
        opacity: 1 !important;
    }

    input,
    textarea,
    div[data-baseweb="input"],
    div[data-baseweb="select"],
    div[data-baseweb="select"] *,
    div[data-baseweb="input"] * {
        color: #0f172a !important;
        background-color: rgba(255,255,255,.92) !important;
        -webkit-text-fill-color: #0f172a !important;
    }

    code, pre {
        color: #0f172a !important;
        background: rgba(226, 232, 240, .55) !important;
    }

    .js-plotly-plot,
    .plot-container,
    .svg-container {
        color: #0f172a !important;
        opacity: 1 !important;
    }

    .main-svg text,
    .main-svg .legendtext,
    .main-svg .gtitle,
    .main-svg .xtitle,
    .main-svg .ytitle,
    .main-svg .xtick text,
    .main-svg .ytick text {
        fill: #0f172a !important;
        color: #0f172a !important;
        opacity: 1 !important;
        font-weight: 650 !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 18px;
        box-shadow: 0 16px 45px rgba(15, 23, 42, 0.055);
        padding: 1rem 1.05rem;
        backdrop-filter: blur(18px);
        color: #0f172a !important;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] p {
        color: #0f172a !important;
        font-weight: 650;
        opacity: 1 !important;
    }

    .hero {
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 26px;
        padding: 1.45rem 1.55rem;
        background:
            linear-gradient(135deg, rgba(255,255,255,.92), rgba(239, 249, 255, .76)),
            linear-gradient(135deg, rgba(37,99,235,.08), rgba(20,184,166,.10));
        box-shadow: var(--shadow);
        backdrop-filter: blur(22px);
        margin-bottom: 1rem;
    }

    .hero-kicker {
        color: var(--cyan);
        font-size: .74rem;
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: .12em;
        margin-bottom: .2rem;
    }

    .hero-title {
        color: var(--ink);
        font-size: 2.05rem;
        line-height: 1.05;
        font-weight: 850;
        letter-spacing: -0.04em;
        margin: 0;
    }

    .hero-subtitle {
        color: #475569;
        font-size: .98rem;
        margin-top: .55rem;
        max-width: 56rem;
    }

    .glass-card {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 20px;
        padding: 1.05rem;
        background: rgba(255,255,255,0.76);
        box-shadow: 0 18px 48px rgba(15, 23, 42, 0.065);
        backdrop-filter: blur(18px);
        margin-bottom: 1rem;
    }

    .mini-label {
        color: #0f172a;
        font-size: .74rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: .09em;
    }

    .big-number {
        color: var(--ink);
        font-size: 1.85rem;
        font-weight: 850;
        letter-spacing: -0.04em;
        line-height: 1;
    }

    .soft-text {
        color: #111827;
        font-size: .9rem;
    }

    .glass-card,
    .glass-card *,
    .decision-row,
    .decision-row * {
        color: #0f172a;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: .28rem .68rem;
        font-size: .76rem;
        font-weight: 800;
        color: #fff !important;
        white-space: nowrap;
        letter-spacing: 0;
    }

    .ghost-badge {
        display: inline-flex;
        border-radius: 999px;
        padding: .25rem .62rem;
        color: #0369a1;
        background: rgba(14, 165, 233, .10);
        border: 1px solid rgba(14, 165, 233, .22);
        font-size: .76rem;
        font-weight: 750;
        margin-right: .3rem;
        margin-bottom: .25rem;
    }

    .pipeline {
        display: grid;
        grid-template-columns: repeat(8, minmax(92px, 1fr));
        gap: .55rem;
        margin: .75rem 0 1.1rem;
    }

    .agent-node {
        position: relative;
        min-height: 122px;
        border-radius: 18px;
        padding: .85rem .72rem;
        border: 1px solid rgba(148,163,184,.24);
        background: rgba(255,255,255,.78);
        box-shadow: 0 14px 38px rgba(15,23,42,.06);
        overflow: hidden;
    }

    .agent-node:after {
        content: "";
        position: absolute;
        right: -11px;
        top: 50%;
        width: 18px;
        height: 2px;
        background: rgba(14,165,233,.45);
    }

    .agent-node:last-child:after {
        display: none;
    }

    .agent-letter {
        width: 30px;
        height: 30px;
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #fff !important;
        font-weight: 850;
        margin-bottom: .5rem;
    }

    .agent-title {
        color: #0f172a !important;
        font-size: .82rem;
        font-weight: 800;
        line-height: 1.15;
        min-height: 2rem;
    }

    .agent-meta {
        color: #0f172a !important;
        font-size: .74rem;
        margin-top: .35rem;
    }

    .status-dot {
        display: inline-block;
        width: .5rem;
        height: .5rem;
        border-radius: 50%;
        margin-right: .32rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: .35rem;
        border-bottom: 1px solid rgba(148,163,184,.25);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px 12px 0 0;
        color: #475569;
        font-weight: 750;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(255,255,255,.78);
        color: #0369a1;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(148,163,184,.22);
        box-shadow: 0 14px 36px rgba(15,23,42,.04);
    }

    .decision-row {
        border-left: 3px solid #06b6d4;
        padding: .55rem .8rem;
        background: rgba(255,255,255,.66);
        border-radius: 0 12px 12px 0;
        margin-bottom: .55rem;
    }

    .info-panel {
        border: 1px solid rgba(148, 163, 184, .22);
        border-radius: 16px;
        background: rgba(255, 255, 255, .78);
        box-shadow: 0 12px 32px rgba(15, 23, 42, .045);
        overflow: hidden;
        margin-top: .65rem;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        padding: .68rem .82rem;
        border-bottom: 1px solid rgba(148, 163, 184, .16);
    }

    .info-row:last-child {
        border-bottom: 0;
    }

    .info-key {
        color: #475569 !important;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .04em;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .info-value {
        color: #0f172a !important;
        font-size: .9rem;
        font-weight: 750;
        text-align: right;
        overflow-wrap: anywhere;
    }

    @media (max-width: 1100px) {
        .pipeline {
            grid-template-columns: repeat(4, minmax(92px, 1fr));
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_yaml(path: str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {}
    with target.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {}
    with target.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def read_text(path: str) -> str:
    target = Path(path)
    if not target.is_file():
        return ""
    return target.read_text(encoding="utf-8")


def clear_cache() -> None:
    read_yaml.clear()
    read_json.clear()
    read_text.clear()
    load_claims.clear()
    load_runs.clear()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_money(value: Any) -> str:
    return f"${as_float(value):,.0f}"


def as_percent(value: Any) -> str:
    return f"{as_float(value) * 100:.1f}%"


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def badge(label: str, color: str) -> str:
    return f'<span class="badge" style="background:{color};">{label}</span>'


def routing_badge(value: str | None) -> str:
    routing = value or "not_run"
    label = routing.replace("_", " ").title()
    return badge(label, ROUTING_COLORS.get(routing, "#64748b"))


def severity_badges(findings: list[dict[str, Any]]) -> str:
    counts = Counter(str(item.get("severity", "info")) for item in findings)
    if not counts:
        return '<span class="ghost-badge">No findings</span>'
    return "".join(
        f'<span class="ghost-badge">{key}: {counts[key]}</span>'
        for key in ("critical", "high", "medium", "low", "info")
        if counts.get(key)
    )


def card(title: str, value: str, subtitle: str = "", accent: str = "#06b6d4") -> None:
    st.markdown(
        f"""
        <div class="glass-card">
          <div class="mini-label">{title}</div>
          <div class="big-number" style="color:{accent};">{value}</div>
          <div class="soft-text">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_panel(items: dict[str, Any]) -> None:
    rows = []
    for key, value in items.items():
        label = str(key).replace("_", " ").title()
        if value is None or value == "":
            display = "Not available"
        elif isinstance(value, bool):
            display = "Yes" if value else "No"
        elif isinstance(value, (int, float)):
            display = f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
        else:
            display = str(value)
        rows.append(
            '<div class="info-row">'
            f'<div class="info-key">{escape(label)}</div>'
            f'<div class="info-value">{escape(display)}</div>'
            '</div>'
        )
    st.markdown(
        '<div class="info-panel">' + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, kicker: str = "SICPS Intelligence Fabric") -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">{kicker}</div>
          <h1 class="hero-title">{title}</h1>
          <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_claims(dataset_root: str) -> list[dict[str, Any]]:
    root = Path(dataset_root)
    claims: list[dict[str, Any]] = []
    if not root.is_dir():
        return claims

    for manifest_path in sorted(root.glob("*/manifest.yaml")):
        manifest = read_yaml(str(manifest_path))
        if not manifest:
            continue
        expected = (
            manifest.get("expected_agent_signals", {})
            .get("agent_exception_triage", {})
            or {}
        )
        fraud = manifest.get("fraud_assessment", {}) or {}
        ocr = manifest.get("ocr", {}) or {}
        claims.append(
            {
                "claim_id": str(manifest.get("claim_id", "")),
                "scenario": str(manifest.get("scenario", manifest_path.parent.name)),
                "bundle": manifest_path.parent.name,
                "bundle_path": str(manifest_path.parent),
                "claim_type": str(manifest.get("claim_type", "")),
                "priority": str(manifest.get("priority", "")),
                "expected_routing": str(
                    expected.get("routing", manifest.get("expected_routing", ""))
                ),
                "expected_outcome": str(manifest.get("expected_outcome", "")),
                "manifest": manifest,
                "fraud_score_expected": as_float(fraud.get("fraud_score")),
                "ocr_confidence_expected": as_float(ocr.get("confidence")),
            }
        )
    return claims


@st.cache_data(show_spinner=False)
def load_runs(runs_root: str) -> list[dict[str, Any]]:
    root = Path(runs_root)
    runs: list[dict[str, Any]] = []
    if not root.is_dir():
        return runs

    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        artifacts = {name: run_dir / name for name in ALL_ARTIFACTS}
        context = read_json(str(artifacts["context_packet.json"]))
        summary = read_json(str(artifacts["claim_summary.json"]))
        coverage = read_json(str(artifacts["coverage_result.json"]))
        settlement = read_json(str(artifacts["settlement_calc.json"]))
        fraud = read_json(str(artifacts["fraud_score.json"]))
        approval = read_json(str(artifacts["approval_packet.json"]))
        metrics = read_json(str(artifacts["metrics.json"]))

        findings = []
        for source in (context, summary, coverage, settlement, fraud):
            findings.extend(source.get("findings", []) or [])
        if approval.get("all_findings"):
            findings = approval.get("all_findings", []) or findings

        exceptions = approval.get("exceptions", []) or []
        created_at = context.get("created_at") or approval.get("finalized_at")
        available = [name for name, path in artifacts.items() if path.is_file()]

        runs.append(
            {
                "claim_id": run_dir.name,
                "run_dir": str(run_dir),
                "available_artifacts": available,
                "context": context,
                "summary": summary,
                "coverage": coverage,
                "settlement": settlement,
                "fraud": fraud,
                "approval": approval,
                "metrics": metrics,
                "findings": findings,
                "exceptions": exceptions,
                "created_at": created_at,
                "routing_decision": approval.get("routing_decision", ""),
                "net_settlement": as_float(
                    approval.get("net_settlement", settlement.get("net_settlement", 0.0))
                ),
                "gross_amount": as_float(settlement.get("gross_amount", 0.0)),
                "fraud_score": as_float(fraud.get("fraud_score", 0.0)),
                "ocr_confidence": as_float(summary.get("overall_confidence", 0.0)),
                "coverage_active": coverage.get("coverage_active"),
                "claim_type": context.get("claim_type", summary.get("claim_type", "")),
                "priority": context.get("priority", ""),
            }
        )
    return runs


def merge_claim_run(claims: list[dict[str, Any]], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    run_by_id = {item["claim_id"]: item for item in runs}
    rows: list[dict[str, Any]] = []
    for claim in claims:
        run = run_by_id.get(claim["claim_id"], {})
        manifest = claim["manifest"]
        rows.append(
            {
                **claim,
                "run": run,
                "has_run": bool(run),
                "routing": run.get("routing_decision") or claim.get("expected_routing", ""),
                "net_settlement": run.get(
                    "net_settlement",
                    as_float((manifest.get("financials") or {}).get("net_settlement")),
                ),
                "fraud_score": run.get("fraud_score", claim.get("fraud_score_expected", 0.0)),
                "ocr_confidence": run.get(
                    "ocr_confidence",
                    claim.get("ocr_confidence_expected", 0.0),
                ),
                "findings": run.get("findings", []),
                "exceptions": run.get("exceptions", []),
            }
        )
    return rows


def plot_layout(fig: go.Figure, height: int = 340) -> go.Figure:
    dark = "#0f172a"
    muted = "#334155"
    grid = "rgba(15, 23, 42, 0.18)"

    fig.update_layout(
        height=height,
        margin=dict(l=24, r=24, t=44, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Inter, sans-serif", color=dark, size=13),
        title=dict(font=dict(color=dark, size=18), x=0.02, xanchor="left"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=dark, size=13),
            bgcolor="rgba(255,255,255,0.55)",
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="rgba(15,23,42,.18)",
            font=dict(color=dark, family="Inter, sans-serif"),
        ),
        uniformtext=dict(mode="show", minsize=11),
    )
    fig.update_xaxes(
        title_font=dict(color=dark, size=14),
        tickfont=dict(color=muted, size=12),
        color=dark,
        gridcolor=grid,
        zerolinecolor=grid,
        linecolor="rgba(15,23,42,.35)",
    )
    fig.update_yaxes(
        title_font=dict(color=dark, size=14),
        tickfont=dict(color=muted, size=12),
        color=dark,
        gridcolor=grid,
        zerolinecolor=grid,
        linecolor="rgba(15,23,42,.35)",
    )
    fig.update_traces(
        textfont=dict(color=dark),
        selector=lambda trace: hasattr(trace, "textfont"),
    )
    return fig


def fraud_gauge(score: float, title: str = "Fraud Risk Gauge") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "font": {"size": 34, "color": "#0f172a"}},
            title={"text": title, "font": {"size": 16, "color": "#334155"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": "#06b6d4"},
                "bgcolor": "rgba(255,255,255,.55)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(20,184,166,.20)"},
                    {"range": [35, 70], "color": "rgba(245,158,11,.22)"},
                    {"range": [70, 100], "color": "rgba(220,38,38,.20)"},
                ],
                "threshold": {
                    "line": {"color": "#dc2626", "width": 4},
                    "thickness": 0.75,
                    "value": 75,
                },
            },
        )
    )
    return plot_layout(fig, 320)


def settlement_distribution(rows: list[dict[str, Any]]) -> go.Figure:
    values = [as_float(row.get("net_settlement")) for row in rows if as_float(row.get("net_settlement")) > 0]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values or [0],
            nbinsx=12,
            marker=dict(color="rgba(6,182,212,.78)", line=dict(color="#0891b2", width=1)),
            hovertemplate="$%{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(title="Settlement Distribution", xaxis_title="Net settlement", yaxis_title="Claims")
    return plot_layout(fig)


def claim_volume_timeline(rows: list[dict[str, Any]]) -> go.Figure:
    counts: Counter[str] = Counter()
    for row in rows:
        manifest = row.get("manifest", {}) or {}
        incident_date = (manifest.get("incident") or {}).get("date")
        dt = parse_dt(incident_date)
        key = dt.strftime("%Y-%m-%d") if dt else "unknown"
        counts[key] += 1
    ordered = sorted(counts.items())
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[item[0] for item in ordered],
            y=[item[1] for item in ordered],
            mode="lines+markers",
            line=dict(color="#2563eb", width=3, shape="spline"),
            marker=dict(size=9, color="#14b8a6"),
            fill="tozeroy",
            fillcolor="rgba(37,99,235,.10)",
        )
    )
    fig.update_layout(title="Claim Volume Timeline", xaxis_title="Incident date", yaxis_title="Claims")
    return plot_layout(fig)


def findings_by_severity(runs: list[dict[str, Any]]) -> go.Figure:
    counts: Counter[str] = Counter()
    for run in runs:
        for finding in run.get("findings", []) or []:
            counts[str(finding.get("severity", "info"))] += 1
    order = ["critical", "high", "medium", "low", "info"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=order,
            y=[counts.get(key, 0) for key in order],
            marker_color=[SEVERITY_COLORS[key] for key in order],
            text=[counts.get(key, 0) for key in order],
            textposition="outside",
        )
    )
    fig.update_layout(title="Findings by Severity", xaxis_title="", yaxis_title="Findings")
    return plot_layout(fig)


def agent_performance_matrix(runs: list[dict[str, Any]]) -> go.Figure:
    agent_labels = [f"Agent {letter}" for letter, _, _, _ in AGENT_PIPELINE]
    z: list[list[float]] = []
    for run in runs:
        run_dir = Path(run["run_dir"])
        row = []
        for _, _, artifact, agent_key in AGENT_PIPELINE:
            artifact_present = (run_dir / artifact).is_file()
            agent_findings = [
                finding
                for finding in run.get("findings", []) or []
                if finding.get("agent") == agent_key
            ]
            score = 100 if artifact_present else 0
            score -= min(len(agent_findings) * 12, 48)
            row.append(max(score, 0))
        z.append(row)
    y = [run["claim_id"] for run in runs] or ["no runs"]
    if not z:
        z = [[0 for _ in agent_labels]]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=agent_labels,
            y=y,
            colorscale=[[0, "#fecaca"], [0.55, "#fde68a"], [1, "#99f6e4"]],
            colorbar=dict(title="Health"),
        )
    )
    fig.update_layout(title="Agent Performance Matrix", xaxis_title="", yaxis_title="")
    return plot_layout(fig, 430)


def coverage_pie(runs: list[dict[str, Any]]) -> go.Figure:
    labels = ["Approved", "Denied", "Review / Pending"]
    approved = sum(1 for run in runs if run.get("coverage_active") is True)
    denied = sum(1 for run in runs if run.get("coverage_active") is False)
    pending = max(len(runs) - approved - denied, 0)
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=[approved, denied, pending],
            hole=0.55,
            marker=dict(colors=["#14b8a6", "#dc2626", "#f59e0b"]),
            textinfo="label+percent",
        )
    )
    fig.update_layout(title="Coverage Approval Mix")
    return plot_layout(fig)


def risk_heatmap(rows: list[dict[str, Any]]) -> go.Figure:
    claim_types = sorted({row.get("claim_type") or "unknown" for row in rows}) or ["unknown"]
    priorities = sorted({row.get("priority") or "unknown" for row in rows}) or ["unknown"]
    matrix = []
    for claim_type in claim_types:
        line = []
        for priority in priorities:
            subset = [
                row
                for row in rows
                if (row.get("claim_type") or "unknown") == claim_type
                and (row.get("priority") or "unknown") == priority
            ]
            avg = sum(as_float(row.get("fraud_score")) for row in subset) / len(subset) if subset else 0
            line.append(round(avg * 100, 1))
        matrix.append(line)
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=priorities,
            y=claim_types,
            colorscale=[[0, "#e0f2fe"], [0.5, "#67e8f9"], [1, "#dc2626"]],
            colorbar=dict(title="Risk %"),
            hovertemplate="Type %{y}<br>Priority %{x}<br>Risk %{z}%<extra></extra>",
        )
    )
    fig.update_layout(title="Risk Heatmap")
    return plot_layout(fig)


def processing_funnel(runs: list[dict[str, Any]]) -> go.Figure:
    stages = []
    for letter, name, artifact, _ in AGENT_PIPELINE:
        count = sum(1 for run in runs if (Path(run["run_dir"]) / artifact).is_file())
        stages.append((f"{letter}. {name}", count))
    fig = go.Figure(
        go.Funnel(
            y=[item[0] for item in stages],
            x=[item[1] for item in stages],
            marker=dict(color=["#2563eb", "#0891b2", "#14b8a6", "#22c55e", "#f59e0b", "#8b5cf6", "#6366f1", "#0f172a"]),
            textinfo="value+percent initial",
        )
    )
    fig.update_layout(title="Processing Funnel")
    return plot_layout(fig, 450)


def provider_network(policy: dict[str, Any], runs: list[dict[str, Any]]) -> go.Figure:
    fraud_cfg = policy.get("fraud", {}) or {}
    rings = fraud_cfg.get("known_fraud_rings", []) or []
    blacklist = fraud_cfg.get("blacklisted_providers", []) or []

    nodes: list[dict[str, Any]] = []
    edges: list[tuple[str, str]] = []

    for ring in rings:
        ring_id = str(ring.get("ring_id"))
        nodes.append({"id": ring_id, "label": ring_id, "kind": "ring"})
        for provider_id in ring.get("provider_ids", []) or []:
            nodes.append({"id": str(provider_id), "label": str(provider_id), "kind": "provider"})
            edges.append((ring_id, str(provider_id)))

    for provider in blacklist:
        provider_id = str(provider.get("id"))
        nodes.append({"id": provider_id, "label": provider_id, "kind": "blacklist"})

    for run in runs:
        provider = (run.get("context") or {}).get("provider") or {}
        provider_id = provider.get("provider_id")
        if provider_id:
            nodes.append({"id": str(provider_id), "label": str(provider_id), "kind": "claim_provider"})
            edges.append((run["claim_id"], str(provider_id)))
            nodes.append({"id": run["claim_id"], "label": run["claim_id"], "kind": "claim"})

    unique: dict[str, dict[str, Any]] = {node["id"]: node for node in nodes if node.get("id")}
    node_ids = list(unique)
    total = max(len(node_ids), 1)
    positions = {
        node_id: (
            math.cos(2 * math.pi * idx / total),
            math.sin(2 * math.pi * idx / total),
        )
        for idx, node_id in enumerate(node_ids)
    }

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in edges:
        if source in positions and target in positions:
            edge_x += [positions[source][0], positions[target][0], None]
            edge_y += [positions[source][1], positions[target][1], None]

    colors = {
        "ring": "#dc2626",
        "blacklist": "#ef4444",
        "provider": "#f59e0b",
        "claim_provider": "#06b6d4",
        "claim": "#2563eb",
    }

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.5, color="rgba(100,116,139,.42)"),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[positions[node_id][0] for node_id in node_ids],
            y=[positions[node_id][1] for node_id in node_ids],
            mode="markers+text",
            text=[unique[node_id]["label"] for node_id in node_ids],
            textposition="top center",
            marker=dict(
                size=[22 if unique[node_id]["kind"] == "ring" else 14 for node_id in node_ids],
                color=[colors.get(unique[node_id]["kind"], "#64748b") for node_id in node_ids],
                line=dict(color="#ffffff", width=2),
            ),
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title="Fraud Ring / Provider Network")
    return plot_layout(fig, 440)


def pipeline_visual(run: dict[str, Any] | None) -> None:
    if not run:
        st.info("Run the selected bundle to populate the agent workflow.")
        return

    run_dir = Path(run["run_dir"])
    findings = run.get("findings", []) or []
    routing = str(run.get("routing_decision") or "")
    html = ['<div class="pipeline">']
    for letter, title, artifact, agent_key in AGENT_PIPELINE:
        present = (run_dir / artifact).is_file()
        agent_findings = [item for item in findings if item.get("agent") == agent_key]
        has_exception = any(item.get("severity") in ("critical", "high") for item in agent_findings)
        needs_review = (
            bool(agent_findings)
            or routing in REVIEW_ROUTINGS
            or routing in {"siu_referral", "coverage_denial"}
        )
        if not present:
            color = "#94a3b8"
            status = "Pending"
        elif has_exception:
            color = "#dc2626"
            status = "Exception"
        elif needs_review and letter in {"E", "F", "G", "H"}:
            color = "#f59e0b"
            status = "Review"
        else:
            color = "#14b8a6"
            status = "Success"
        html.append(
            '<div class="agent-node">'
            f'<div class="agent-letter" style="background:{color};">{letter}</div>'
            f'<div class="agent-title">{title}</div>'
            f'<div class="agent-meta"><span class="status-dot" style="background:{color};"></span>{status}</div>'
            f'<div class="agent-meta">{len(agent_findings)} finding(s)</div>'
            f'<div class="agent-meta">{artifact}</div>'
            '</div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def selected_claim_selector(rows: list[dict[str, Any]], label: str = "Claim") -> dict[str, Any]:
    options = {f'{row["claim_id"]} | {row["scenario"]}': row for row in rows}
    selected = st.selectbox(label, list(options), label_visibility="collapsed")
    return options[selected]


def run_selected_claim(row: dict[str, Any], config_path: str, runs_root: str, force: bool) -> None:
    with st.spinner("Executing SICPS agent fabric"):
        runner = PipelineRunner(config_path=config_path, runs_dir=runs_root)
        result = runner.run(row["bundle_path"], force=force)
        clear_cache()
    st.success(
        f"{result.claim_id} resolved as {result.routing_decision} with net {as_money(result.net_settlement)}"
    )
    st.rerun()


def claim_profile(row: dict[str, Any]) -> None:
    manifest = row.get("manifest", {}) or {}
    run = row.get("run", {}) or {}
    context = run.get("context", {}) if run else {}
    coverage = run.get("coverage", {}) if run else {}
    settlement = run.get("settlement", {}) if run else {}
    fraud = run.get("fraud", {}) if run else {}
    approval = run.get("approval", {}) if run else {}

    claimant = context.get("claimant") or manifest.get("claimant", {}) or {}
    policy = context.get("policy") or manifest.get("policy", {}) or {}
    incident = context.get("incident") or manifest.get("incident", {}) or {}
    provider = context.get("provider") or manifest.get("provider", {}) or {}

    st.markdown(routing_badge(approval.get("routing_decision") or row.get("routing")), unsafe_allow_html=True)
    cols = st.columns(5)
    cols[0].metric("Net Settlement", as_money(row.get("net_settlement")))
    cols[1].metric("Fraud Score", as_percent(row.get("fraud_score")))
    cols[2].metric("OCR Confidence", as_percent(row.get("ocr_confidence")))
    cols[3].metric("Findings", len(row.get("findings", []) or []))
    cols[4].metric("Exceptions", len(row.get("exceptions", []) or []))

    st.markdown(severity_badges(row.get("findings", []) or []), unsafe_allow_html=True)

    left, mid, right = st.columns(3)
    with left:
        card("Claim Profile", claimant.get("full_name", "Unknown"), f'{row["claim_type"]} | {row["priority"]}', "#2563eb")
        info_panel(
            {
                "claim_id": row["claim_id"],
                "policy_number": claimant.get("policy_number") or policy.get("policy_number"),
                "prior_claims_18mo": claimant.get("prior_claims_18mo"),
                "new_claimant": claimant.get("new_claimant"),
            }
        )
    with mid:
        coverage_text = "Active" if coverage.get("coverage_active") is True else "Denied" if coverage.get("coverage_active") is False else "Pending"
        card("Coverage Result", coverage_text, coverage.get("denial_reason") or policy.get("policy_type", ""), "#14b8a6")
        info_panel(
            {
                "state": coverage.get("applicable_state") or policy.get("state"),
                "deductible": coverage.get("deductible") or policy.get("deductible"),
                "late_notice_violation": coverage.get("late_notice_violation"),
                "policy_expired": coverage.get("policy_expired"),
            }
        )
    with right:
        risk = fraud.get("risk_level", "pending")
        card("Fraud Result", str(risk).title(), f'SIU: {fraud.get("siu_referral", False)}', "#06b6d4")
        info_panel(
            {
                "provider_id": provider.get("provider_id"),
                "provider_name": provider.get("provider_name"),
                "blacklisted": fraud.get("provider_blacklisted", provider.get("is_blacklisted")),
                "fraud_ring": fraud.get("provider_ring_id", provider.get("fraud_ring_id")),
            }
        )

    st.subheader("Agent Pipeline")
    pipeline_visual(run if run else None)

    detail_tabs = st.tabs(["Decision Trail", "Findings Timeline", "Artifacts"])
    with detail_tabs[0]:
        decision_rows = [
            ("FNOL Intake", "context_packet.json", bool(context)),
            ("Coverage", coverage.get("denial_reason") or str(coverage.get("coverage_active", "pending")), bool(coverage)),
            ("Settlement", as_money(settlement.get("net_settlement", 0.0)), bool(settlement)),
            ("Fraud", f'{fraud.get("risk_level", "pending")} / SIU {fraud.get("siu_referral", False)}', bool(fraud)),
            ("Final Routing", approval.get("routing_decision", "pending"), bool(approval)),
        ]
        for title, value, done in decision_rows:
            color = "#14b8a6" if done else "#94a3b8"
            st.markdown(
                f"""
                <div class="decision-row" style="border-left-color:{color};">
                  <strong>{title}</strong><br>
                  <span class="soft-text">{value}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with detail_tabs[1]:
        findings = row.get("findings", []) or []
        if findings:
            st.dataframe(
                [
                    {
                        "timestamp": item.get("timestamp"),
                        "agent": item.get("agent"),
                        "severity": item.get("severity"),
                        "finding": item.get("finding_id"),
                        "recommendation": item.get("recommendation"),
                    }
                    for item in findings
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No findings recorded for this claim.")
    with detail_tabs[2]:
        if run:
            artifact = st.selectbox("Artifact", run.get("available_artifacts", []))
            path = Path(run["run_dir"]) / artifact
            if artifact.endswith(".json"):
                st.json(read_json(str(path)), expanded=False)
            elif artifact.endswith(".csv"):
                st.code(read_text(str(path)), language="csv")
            else:
                st.markdown(read_text(str(path)))
        else:
            st.info("No run artifacts available.")


def executive_dashboard(rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> None:
    hero(
        "Enterprise Claims Intelligence",
        "A premium AI operating layer for intake, coverage, settlement, fraud intelligence, and audit-ready insurance decisions.",
    )

    total = len(rows)
    approved = sum(1 for row in rows if row.get("routing") in AUTO_ROUTINGS)
    denied = sum(1 for row in rows if row.get("routing") in DENIAL_ROUTINGS)
    pending = sum(1 for row in rows if row.get("routing") in REVIEW_ROUTINGS)
    fraud_risk = sum(1 for row in rows if as_float(row.get("fraud_score")) >= 0.4 or row.get("routing") == "siu_referral")
    avg_settlement = sum(as_float(row.get("net_settlement")) for row in rows) / total if total else 0
    avg_ocr = sum(as_float(row.get("ocr_confidence")) for row in rows) / total if total else 0
    accuracy_values = [as_float((run.get("metrics") or {}).get("extraction_accuracy")) for run in runs if run.get("metrics")]
    accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else avg_ocr

    cols = st.columns(4)
    cols[0].metric("Total Claims", total, "+12 scenario corpus")
    cols[1].metric("Approved Claims", approved, "straight-through")
    cols[2].metric("Denied Claims", denied, "coverage terminal")
    cols[3].metric("Pending Reviews", pending, "human review")
    cols = st.columns(4)
    cols[0].metric("Fraud Risk Claims", fraud_risk, "SIU/elevated")
    cols[1].metric("Average Settlement", as_money(avg_settlement), "net payable")
    cols[2].metric("Processing Accuracy", as_percent(accuracy), "extraction confidence")
    cols[3].metric("OCR Confidence", as_percent(avg_ocr), "document quality")

    left, right = st.columns([1.15, 1])
    with left:
        st.plotly_chart(settlement_distribution(rows), use_container_width=True)
    with right:
        max_risk = max([as_float(row.get("fraud_score")) for row in rows] or [0])
        st.plotly_chart(fraud_gauge(max_risk, "Highest Active Fraud Risk"), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(claim_volume_timeline(rows), use_container_width=True)
    with right:
        st.plotly_chart(coverage_pie(runs), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(findings_by_severity(runs), use_container_width=True)
    with right:
        st.plotly_chart(processing_funnel(runs), use_container_width=True)


def claims_processing(rows: list[dict[str, Any]], config_path: str, runs_root: str, force: bool) -> None:
    hero(
        "Claims Processing",
        "Operate the SICPS multi-agent pipeline claim by claim, with live workflow status and artifact-level traceability.",
        "Claims Command Center",
    )
    selected = selected_claim_selector(rows)
    run_col, refresh_col = st.columns([1, 5])
    with run_col:
        if st.button("Run Claim", type="primary", use_container_width=True):
            run_selected_claim(selected, config_path, runs_root, force)
    with refresh_col:
        st.caption("Runs are executed by the existing `PipelineRunner` and persisted to `runs/{claim_id}`.")

    claim_profile(selected)

    st.subheader("Portfolio Queue")
    st.dataframe(
        [
            {
                "Claim": row["claim_id"],
                "Scenario": row["scenario"],
                "Type": row["claim_type"],
                "Priority": row["priority"],
                "Routing": row["routing"],
                "Net": row["net_settlement"],
                "Fraud": row["fraud_score"],
                "OCR": row["ocr_confidence"],
                "Run": row["has_run"],
            }
            for row in rows
        ],
        use_container_width=True,
        hide_index=True,
    )


def coverage_intelligence(rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> None:
    hero(
        "Coverage Intelligence",
        "Policy status, late notice, exclusions, expiration windows, and denial reasoning from coverage validation artifacts.",
        "Coverage Graph",
    )
    st.plotly_chart(coverage_pie(runs), use_container_width=True)
    coverage_rows = []
    for row in rows:
        run = row.get("run", {}) or {}
        coverage = run.get("coverage", {}) or {}
        manifest_policy = (row.get("manifest", {}) or {}).get("policy", {}) or {}
        coverage_rows.append(
            {
                "Claim": row["claim_id"],
                "Type": row["claim_type"],
                "Coverage Active": coverage.get("coverage_active"),
                "Policy Expired": coverage.get("policy_expired", manifest_policy.get("policy_expired")),
                "Late Notice": coverage.get("late_notice_violation"),
                "Denial Reason": coverage.get("denial_reason"),
                "Deductible": coverage.get("deductible", manifest_policy.get("deductible")),
                "Limit": coverage.get("coverage_limit", manifest_policy.get("coverage_limit")),
            }
        )
    st.dataframe(coverage_rows, use_container_width=True, hide_index=True)


def settlement_analytics(rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> None:
    hero(
        "Settlement Analytics",
        "Financial settlement intelligence across gross loss, deductibles, total loss detection, and variance signals.",
        "Settlement Studio",
    )
    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(settlement_distribution(rows), use_container_width=True)
    with right:
        totals = [
            run.get("settlement", {}) or {}
            for run in runs
            if run.get("settlement")
        ]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[run["claim_id"] for run in runs],
                y=[as_float(run.get("settlement", {}).get("gross_amount")) for run in runs],
                name="Gross",
                marker_color="#93c5fd",
            )
        )
        fig.add_trace(
            go.Bar(
                x=[run["claim_id"] for run in runs],
                y=[as_float(run.get("settlement", {}).get("net_settlement")) for run in runs],
                name="Net",
                marker_color="#14b8a6",
            )
        )
        fig.update_layout(title="Gross vs Net Settlement", barmode="group", xaxis_title="", yaxis_title="Amount")
        st.plotly_chart(plot_layout(fig), use_container_width=True)

    st.dataframe(
        [
            {
                "Claim": run["claim_id"],
                "Gross": run.get("settlement", {}).get("gross_amount"),
                "Deductible": run.get("settlement", {}).get("deductible"),
                "Net": run.get("settlement", {}).get("net_settlement"),
                "Total Loss": run.get("settlement", {}).get("total_loss_flag"),
                "Repair Variance": run.get("settlement", {}).get("repair_variance_pct"),
                "Medical Variance": run.get("settlement", {}).get("medical_variance_pct"),
            }
            for run in runs
            if run.get("settlement")
        ],
        use_container_width=True,
        hide_index=True,
    )


def fraud_center(rows: list[dict[str, Any]], runs: list[dict[str, Any]], policy: dict[str, Any]) -> None:
    hero(
        "AI Fraud Intelligence Center",
        "SIU referral intelligence, provider risk, prior claim pressure, and fraud-ring topology grounded in actual fraud_score artifacts.",
        "Fraud Operations",
    )
    selected = selected_claim_selector(rows, "Fraud claim")
    run = selected.get("run", {}) or {}
    fraud = run.get("fraud", {}) or {}
    manifest = selected.get("manifest", {}) or {}
    provider = (run.get("context", {}) or {}).get("provider") or manifest.get("provider", {}) or {}
    claimant = (run.get("context", {}) or {}).get("claimant") or manifest.get("claimant", {}) or {}

    left, mid, right = st.columns([1, 1, 1])
    with left:
        st.plotly_chart(fraud_gauge(as_float(selected.get("fraud_score")), "Selected Claim Fraud Score"), use_container_width=True)
    with mid:
        card("Risk Classification", str(fraud.get("risk_level", "pending")).title(), f'Claim {selected["claim_id"]}', "#dc2626" if fraud.get("siu_referral") else "#06b6d4")
        card("SIU Referral", str(fraud.get("siu_referral", False)), "terminal hold when true", "#dc2626" if fraud.get("siu_referral") else "#14b8a6")
    with right:
        card("Provider Risk", provider.get("provider_id", "Unknown"), provider.get("provider_name", ""), "#f59e0b")
        card("Prior Claims", str(claimant.get("prior_claims_18mo", 0)), "18 month window", "#2563eb")

    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(provider_network(policy, runs), use_container_width=True)
    with right:
        st.plotly_chart(risk_heatmap(rows), use_container_width=True)

    indicators = fraud.get("indicators", []) or []
    st.subheader("Fraud Indicators")
    if indicators:
        st.dataframe(indicators, use_container_width=True, hide_index=True)
    else:
        st.info("No fraud indicators found for this claim.")

    st.subheader("Provider Risk Analysis")
    st.dataframe(
        [
            {
                "Claim": run["claim_id"],
                "Provider": ((run.get("context", {}) or {}).get("provider") or {}).get("provider_id"),
                "Blacklisted": (run.get("fraud", {}) or {}).get("provider_blacklisted"),
                "Ring Match": (run.get("fraud", {}) or {}).get("provider_ring_match"),
                "Ring ID": (run.get("fraud", {}) or {}).get("provider_ring_id"),
                "Fraud Score": run.get("fraud_score"),
            }
            for run in runs
            if run.get("fraud")
        ],
        use_container_width=True,
        hide_index=True,
    )


def audit_compliance(rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> None:
    hero(
        "Audit & Compliance",
        "A regulator-ready evidence plane for agent findings, exceptions, coverage decisions, fraud decisions, and final audit logs.",
        "Compliance Workspace",
    )
    findings = []
    for run in runs:
        for item in run.get("findings", []) or []:
            findings.append({"claim_id": run["claim_id"], **item})

    agents = sorted({item.get("agent", "") for item in findings if item.get("agent")})
    severities = sorted({item.get("severity", "") for item in findings if item.get("severity")})
    claim_ids = sorted({item.get("claim_id", "") for item in findings if item.get("claim_id")})

    cols = st.columns(4)
    severity_filter = cols[0].multiselect("Severity", severities, default=severities)
    agent_filter = cols[1].multiselect("Agent", agents, default=agents)
    claim_filter = cols[2].multiselect("Claim ID", claim_ids, default=claim_ids)
    date_query = cols[3].text_input("Date contains", value="")

    filtered = [
        item
        for item in findings
        if item.get("severity") in severity_filter
        and item.get("agent") in agent_filter
        and item.get("claim_id") in claim_filter
        and (not date_query or date_query in str(item.get("timestamp", "")))
    ]

    st.plotly_chart(findings_by_severity(runs), use_container_width=True)
    st.dataframe(
        [
            {
                "Claim": item.get("claim_id"),
                "Timestamp": item.get("timestamp"),
                "Agent": item.get("agent"),
                "Severity": item.get("severity"),
                "Finding": item.get("finding_id"),
                "Recommendation": item.get("recommendation"),
                "Human Review": item.get("requires_human_review"),
            }
            for item in filtered
        ],
        use_container_width=True,
        hide_index=True,
    )

    selected_run_ids = [run["claim_id"] for run in runs]
    if selected_run_ids:
        selected_claim = st.selectbox("Audit trail", selected_run_ids)
        run = next(run for run in runs if run["claim_id"] == selected_claim)
        run_dir = Path(run["run_dir"])
        audit_path = run_dir / "audit_log.md"
        exceptions_path = run_dir / "exceptions.md"
        left, right = st.columns(2)
        with left:
            st.subheader("Audit Trail")
            st.markdown(read_text(str(audit_path)) or "No audit log available.")
        with right:
            st.subheader("Exceptions")
            st.markdown(read_text(str(exceptions_path)) or "No exception report available.")


def system_monitoring(runs: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    hero(
        "System Monitoring",
        "Operational health for the multi-agent pipeline: throughput, artifact completion, errors, execution coverage, and OCR quality.",
        "Reliability Command",
    )
    complete_runs = sum(1 for run in runs if "approval_packet.json" in run.get("available_artifacts", []))
    error_count = sum(1 for row in rows if row.get("routing") in {"siu_referral", "coverage_denial"})
    avg_ocr = sum(as_float(row.get("ocr_confidence")) for row in rows) / len(rows) if rows else 0
    avg_duration = sum(as_float((run.get("metrics") or {}).get("pipeline_duration_seconds")) for run in runs) / len(runs) if runs else 0

    cols = st.columns(4)
    cols[0].metric("Completed Runs", complete_runs)
    cols[1].metric("Pipeline Throughput", f"{complete_runs}/{len(rows)}")
    cols[2].metric("Error / Hold Count", error_count)
    cols[3].metric("Avg Processing Time", f"{avg_duration:.2f}s")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(agent_performance_matrix(runs), use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_trace(
            go.Box(
                y=[as_float(row.get("ocr_confidence")) * 100 for row in rows],
                name="OCR Quality",
                marker_color="#06b6d4",
                boxmean=True,
            )
        )
        fig.update_layout(title="OCR Quality Metrics", yaxis_title="Confidence %")
        st.plotly_chart(plot_layout(fig), use_container_width=True)

    st.subheader("Artifact Completion")
    st.dataframe(
        [
            {
                "Claim": run["claim_id"],
                **{artifact: artifact in run.get("available_artifacts", []) for artifact in JSON_ARTIFACTS},
            }
            for run in runs
        ],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    with st.sidebar:
        st.markdown("### SICPS Intelligence")
        st.caption("AI-powered claims operating fabric")
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.divider()
        dataset_root = st.text_input("Dataset root", value=str(DEFAULT_DATASET))
        runs_root = st.text_input("Runs root", value=str(DEFAULT_RUNS))
        config_path = st.text_input("Policy pack", value=str(DEFAULT_CONFIG))
        force = st.toggle("Force recompute", value=False)
        if st.button("Refresh workspace", use_container_width=True):
            clear_cache()
            st.rerun()
        st.divider()
        st.caption("Existing artifact contract")
        for name in JSON_ARTIFACTS:
            st.markdown(f'<span class="ghost-badge">{name}</span>', unsafe_allow_html=True)

    claims = load_claims(dataset_root)
    runs = load_runs(runs_root)
    policy = read_yaml(config_path)

    if not claims:
        hero("Workspace Not Found", "No claim manifests were found under the configured dataset root.")
        return
    if not Path(config_path).is_file():
        hero("Policy Pack Missing", f"The configured policy file does not exist: {config_path}")
        return

    rows = merge_claim_run(claims, runs)

    if page == "Executive Dashboard":
        executive_dashboard(rows, runs)
    elif page == "Claims Processing":
        claims_processing(rows, config_path, runs_root, force)
    elif page == "Coverage Intelligence":
        coverage_intelligence(rows, runs)
    elif page == "Settlement Analytics":
        settlement_analytics(rows, runs)
    elif page == "Fraud Intelligence Center":
        fraud_center(rows, runs, policy)
    elif page == "Audit & Compliance":
        audit_compliance(rows, runs)
    elif page == "System Monitoring":
        system_monitoring(runs, rows)


if __name__ == "__main__":
    main()
