"""
Simulation/Mesh AI Analysis — Streamlit Dashboard
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from simscale_log_parser import parse_log_file

# ── Constants ──────────────────────────────────────────────────────────────────

CSV_PATH = Path("mesh_dataset.csv")
NON_FEATURE_COLS = {"run_id", "label", "final_Cl", "final_Cd", "converged"}
LABEL_COLORS = {"good": "#22c55e", "marginal": "#f59e0b", "bad": "#ef4444", "unlabeled": "#94a3b8"}
LABEL_ORDER  = ["good", "marginal", "bad", "unlabeled"]

MESH_THRESHOLDS = {
    "overall_quality":          {"good": 0.75,  "marginal": 0.65,  "lower": False, "unit": "",  "desc": "Overall quality score (0–1). Composite of non-orthogonality, skewness, and aspect ratio."},
    "nonOrthogonality_average": {"good": 30.0,  "marginal": 45.0,  "lower": True,  "unit": "°", "desc": "Mean non-orthogonality angle. Measures how much face normals deviate from the cell-centre vector."},
    "nonOrthogonality_p99_9":   {"good": 70.0,  "marginal": 85.0,  "lower": True,  "unit": "°", "desc": "99.9th-percentile non-orthogonality. Captures worst cells without outlier distortion."},
    "skewness_average":         {"good": 0.5,   "marginal": 1.0,   "lower": True,  "unit": "",  "desc": "Average skewness. Measures face centroid deviation from the cell-centre connector line."},
    "skewness_max":             {"good": 2.0,   "marginal": 4.0,   "lower": True,  "unit": "",  "desc": "Maximum skewness. Values > 4 can cause solver divergence."},
    "aspectRatio_average":      {"good": 3.0,   "marginal": 10.0,  "lower": True,  "unit": "",  "desc": "Average aspect ratio (longest/shortest edge). High values in the bulk degrade accuracy."},
    "tetAspectRatio_max":       {"good": 20.0,  "marginal": 100.0, "lower": True,  "unit": "",  "desc": "Maximum tet aspect ratio. Extreme values reduce solver accuracy significantly."},
    "volumeRatio_max":          {"good": 10.0,  "marginal": 50.0,  "lower": True,  "unit": "",  "desc": "Maximum adjacent-cell volume ratio. Abrupt jumps degrade gradient reconstruction."},
}
HIGHLIGHT_METRICS = list(MESH_THRESHOLDS.keys())

RADAR_METRICS = [
    "nonOrthogonality_average", "nonOrthogonality_p99_9",
    "skewness_average", "skewness_max",
    "aspectRatio_average", "tetAspectRatio_average",
    "volumeRatio_average", "tetEdgeRatio_average",
]

CONVERGENCE_THRESHOLDS = {"Converged": 0.01, "Marginal": 0.05}

RESIDUAL_META: dict[str, dict] = {
    "Ux":      {"description": "X-momentum residual",           "interpretation": "Balances forces in the streamwise direction.",                                     "high_risk": "High Ux → x-momentum not balanced; strong flow features, poor mesh, or too few iterations."},
    "Uy":      {"description": "Y-momentum residual",           "interpretation": "Residual for cross-stream / vertical velocity.",                                   "high_risk": "High Uy → cross-stream forces not balanced; check BCs and near-wall mesh."},
    "Uz":      {"description": "Z-momentum residual",           "interpretation": "Residual for spanwise velocity.",                                                  "high_risk": "High Uz → spanwise momentum not converged; may indicate 3-D separation."},
    "p":       {"description": "Pressure / continuity residual","interpretation": "Measures how well mass is conserved across the domain.",                           "high_risk": "High p → mass conservation violated; check inlet/outlet BCs and relaxation."},
    "k":       {"description": "Turbulent kinetic energy (k)",  "interpretation": "Residual of the k transport equation (k-ω / k-ε models).",                        "high_risk": "High k → turbulence unresolved; flow instability or poor turbulence BCs."},
    "omega":   {"description": "Specific dissipation rate (ω)", "interpretation": "Residual of the ω equation in k-ω models.",                                       "high_risk": "High ω → dissipation not converged; review y⁺ and near-wall layers."},
    "epsilon": {"description": "Turbulent dissipation rate (ε)","interpretation": "Residual of the ε equation in k-ε models.",                                       "high_risk": "High ε → dissipation balance not achieved; refine mesh in high-gradient regions."},
    "h":       {"description": "Energy / enthalpy residual",    "interpretation": "Residual of the energy equation. Active only when heat transfer is included.",     "high_risk": "High h → thermal field not converged; review thermal BCs."},
    "T":       {"description": "Temperature residual",          "interpretation": "Residual for the temperature transport equation.",                                  "high_risk": "High T → temperature field not converged; check thermal BCs and fluid properties."},
}

PROBE_META: dict[str, dict] = {
    "domain":  {"title": "Domain Probe",  "description": "Field values at a monitoring point inside the bulk flow.",
                "variables": {"T": {"label":"Temperature","unit":"K","meaning":"Should be constant in isothermal flows. Rising T → heat accumulation or poor thermal BC."}, "Ux": {"label":"X-velocity","unit":"m/s","meaning":"Streamwise velocity. Should plateau when flow is attached; oscillations → unsteady separation."}, "Uy": {"label":"Y-velocity","unit":"m/s","meaning":"Near zero in symmetric flows; persistent non-zero → lateral pressure gradient or separation."}, "Uz": {"label":"Z-velocity","unit":"m/s","meaning":"Near zero in 2-D / symmetric configurations."}, "p": {"label":"Pressure","unit":"Pa","meaning":"Large drifts → pressure field still evolving; run more iterations."}}},
    "inlet":   {"title": "Inlet Probe",   "description": "Field values at the inlet boundary. Should match prescribed BCs.",
                "variables": {"T": {"label":"Temperature","unit":"K","meaning":"Should match inlet BC value; drift → BC not applied correctly."}, "Ux": {"label":"X-velocity","unit":"m/s","meaning":"Should equal prescribed inlet speed. Time-varying Ux → inlet reflection or reversed flow."}, "Uy": {"label":"Y-velocity","unit":"m/s","meaning":"~0 for straight axial inlet; non-zero → swirl or cross-flow at inlet."}, "Uz": {"label":"Z-velocity","unit":"m/s","meaning":"~0; deviation → inlet misalignment or 3-D bleed-in."}, "p": {"label":"Pressure","unit":"Pa","meaning":"Should match prescribed inlet pressure; oscillating → wave reflections or instability."}}},
    "outlet":  {"title": "Outlet Probe",  "description": "Field values at the outlet boundary. Key for backflow and pressure build-up detection.",
                "variables": {"T": {"label":"Temperature","unit":"K","meaning":"Driven by upstream convection. Rising T at outlet → recirculation pulling fluid back."}, "Ux": {"label":"X-velocity","unit":"m/s","meaning":"Positive = correct exit flow. **Negative Ux = backflow** — fluid ingesting through outlet."}, "Uy": {"label":"Y-velocity","unit":"m/s","meaning":"Should be small vs Ux; large Uy → outlet not aligned with primary flow."}, "Uz": {"label":"Z-velocity","unit":"m/s","meaning":"Near zero; significant Uz → undissipated swirl; extend outlet downstream."}, "p": {"label":"Pressure","unit":"Pa","meaning":"Usually reference pressure (0 Pa gauge); oscillating → outlet BC not enforced correctly."}}},
    "wall":    {"title": "Wall Probe",    "description": "Field values at a no-slip wall surface. Velocity must be ~0.",
                "variables": {"T": {"label":"Temperature","unit":"K","meaning":"Should hold at prescribed wall temperature; drifting → incorrect thermal BC."}, "Ux": {"label":"X-velocity","unit":"m/s","meaning":"**Must be ~0 at no-slip wall.** Non-zero → probe off-wall or wrong BC type (slip/symmetry)."}, "Uy": {"label":"Y-velocity","unit":"m/s","meaning":"**Must be ~0 at no-slip wall.** Large values → check probe placement and BC."}, "Uz": {"label":"Z-velocity","unit":"m/s","meaning":"~0 except rotating walls; non-zero → unintentional moving-wall condition."}, "p": {"label":"Pressure","unit":"Pa","meaning":"Surface static pressure used for drag/lift. Oscillations → unsteady separation; check y⁺ and inflation layers."}}},
}


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _quality_label(value: float, metric: str) -> tuple[str, str]:
    """Return (label, css_class) for a metric value vs thresholds."""
    t = MESH_THRESHOLDS.get(metric, {})
    if not t:
        return "—", ""
    g, m, low = t["good"], t["marginal"], t["lower"]
    if low:
        if value <= g:   return "Good",     "status-good"
        if value <= m:   return "Marginal", "status-warn"
        return "Poor",   "status-bad"
    else:
        if value >= g:   return "Good",     "status-good"
        if value >= m:   return "Marginal", "status-warn"
        return "Poor",   "status-bad"


def convergence_status(cv: float) -> tuple[str, str]:
    if cv < CONVERGENCE_THRESHOLDS["Converged"]: return "Converged ✅", "#22c55e"
    if cv < CONVERGENCE_THRESHOLDS["Marginal"]:  return "Marginal ⚠️",  "#f59e0b"
    return "Not converged ❌", "#ef4444"


def _detect_probe_type(filename: str, columns: list[str]) -> str:
    name = filename.lower()
    for kw, pt in [("residual","residuals"),("inlet","inlet"),("outlet","outlet"),("wall","wall"),("domain","domain")]:
        if kw in name:
            return pt
    residual_vars = {"Ux","Uy","Uz","p","k","omega","epsilon","h"}
    if len(residual_vars & set(columns)) >= 3 and "T" not in columns:
        return "residuals"
    return "domain"


def _is_steady(series: pd.Series, window_pct: float = 0.20) -> tuple[bool, float]:
    n = max(2, int(len(series) * window_pct))
    w = series.iloc[-n:]
    mean = w.mean()
    cv = abs(w.std() / mean) if mean != 0 else float("inf")
    return cv < 0.02, cv


def _render_residuals(df: pd.DataFrame, time_col: str) -> None:
    rcols = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
    if not rcols:
        st.warning("No residual columns found.")
        return

    st.subheader("Residual history — log scale")
    st.caption("Residuals must decrease monotonically. Target: all variables below **1×10⁻⁴** for a converged steady-state solution.")

    plot_df = df[[time_col] + rcols].copy()
    for c in rcols:
        plot_df[c] = plot_df[c].abs().clip(lower=1e-20)

    pal = px.colors.qualitative.Plotly
    fig = go.Figure()
    for i, col in enumerate(rcols):
        fig.add_trace(go.Scatter(x=df[time_col], y=plot_df[col], mode="lines", name=col,
                                 line=dict(color=pal[i % len(pal)], width=2),
                                 hovertemplate=f"<b>{col}</b>: %{{y:.3e}}<extra></extra>"))
    fig.add_hline(y=1e-4, line_dash="dot", line_color="#22c55e",
                  annotation_text="1×10⁻⁴  converged", annotation_position="top right")
    fig.add_hline(y=1e-3, line_dash="dot", line_color="#f59e0b",
                  annotation_text="1×10⁻³  marginal",  annotation_position="top right")
    fig.update_layout(yaxis_type="log", yaxis_title="Residual", xaxis_title=time_col,
                      height=440, margin=dict(t=20,b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig, width="stretch")

    st.divider()
    rows = []
    for col in rcols:
        init  = float(df[col].abs().iloc[0])
        final = abs(float(df[col].iloc[-1]))
        orders = max(0, np.log10(init / max(final, 1e-20)))
        try:
            log_v = np.log10(np.clip(df[col].abs().values, 1e-20, None))
            slope = float(np.polyfit(np.arange(len(log_v)), log_v, 1)[0])
        except Exception:
            slope = 0.0
        if   final < 1e-4:       status = "Converged ✅"
        elif final < 1e-3:       status = "Marginal ⚠️"
        elif slope > 0.001:      status = "Diverging ❌"
        elif abs(slope) < 5e-4:  status = "Stalled ⚠️"
        else:                    status = "Decreasing 🔄"
        meta = RESIDUAL_META.get(col, {})
        rows.append({"Variable": col, "Description": meta.get("description","—"),
                     "Initial": init, "Final": final, "Orders dropped": orders, "Status": status})

    def _cr(v):
        if "Converged" in v: return "background-color:#dcfce7;color:#15803d;font-weight:600"
        if "Marginal"  in v or "Stalled" in v: return "background-color:#fef9c3;color:#92400e;font-weight:600"
        if "Diverging" in v: return "background-color:#fee2e2;color:#991b1b;font-weight:600"
        return "background-color:#ede9fe;color:#5b21b6;font-weight:600"

    rtbl = pd.DataFrame(rows)
    st.subheader("Convergence status")
    st.dataframe(rtbl.style.format({"Initial":"{:.3e}","Final":"{:.3e}","Orders dropped":"{:.1f}"})
                 .applymap(_cr, subset=["Status"]), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Diagnosis")
    div_ = [r for r in rows if "Diverging"  in r["Status"]]
    sta_ = [r for r in rows if "Stalled"    in r["Status"]]
    dec_ = [r for r in rows if "Decreasing" in r["Status"]]
    con_ = [r for r in rows if "Converged"  in r["Status"]]
    mar_ = [r for r in rows if "Marginal"   in r["Status"]]
    if len(con_) == len(rows):
        st.success("All residuals < 1×10⁻⁴ — simulation is **fully converged**. Extracted force/field values are reliable.")
    if div_:
        st.error(f"**Diverging**: {', '.join(r['Variable'] for r in div_)}. Residuals rising — reduce relaxation factors or time step and check mesh quality.")
    if sta_:
        st.warning(f"**Stalled**: {', '.join(r['Variable'] for r in sta_)}. Stopped decreasing before 1×10⁻⁴. Causes: (1) physically unsteady flow, (2) coarse mesh in high-gradient regions, (3) relaxation factors too high.")
    if dec_:
        st.info(f"**Still converging**: {', '.join(r['Variable'] for r in dec_)}. On the right track — run more iterations.")
    if mar_:
        st.warning(f"**Marginal**: {', '.join(r['Variable'] for r in mar_)}. Between 1×10⁻⁴ and 1×10⁻³ — acceptable for bulk quantities but not for sensitive forces or heat transfer.")
    with st.expander("Per-variable physics notes"):
        for col in rcols:
            m = RESIDUAL_META.get(col, {})
            if m:
                st.markdown(f"**{col} — {m['description']}**  \n{m['interpretation']}  \n*Risk if high*: {m['high_risk']}")


def _render_field_probe(df: pd.DataFrame, time_col: str, probe_type: str) -> None:
    fcols = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
    if not fcols:
        st.warning("No field columns found.")
        return
    meta = PROBE_META.get(probe_type, PROBE_META["domain"])
    st.subheader(f"{meta['title']} — Time history")
    st.caption(meta["description"])

    n_ss = sum(1 for c in fcols if _is_steady(df[c])[0])
    mc = st.columns(3)
    mc[0].metric("Variables", len(fcols))
    mc[1].metric("At steady state", f"{n_ss}/{len(fcols)}")
    mc[2].metric("Time steps", len(df))

    pal  = px.colors.qualitative.Plotly
    roll = max(3, len(df) // 10)
    gcols = st.columns(min(2, len(fcols)))
    for i, col in enumerate(fcols):
        vm = meta["variables"].get(col, {"label": col, "unit": "", "meaning": ""})
        is_ss, cv = _is_steady(df[col])
        fmean = df[col].iloc[max(0, int(len(df)*0.8)):].mean()
        icon  = "✅" if is_ss else ("⚠️" if cv < 0.10 else "❌")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[time_col], y=df[col], mode="lines", name="Raw",
                                 line=dict(color=pal[i%len(pal)], width=1.2), opacity=0.45,
                                 hovertemplate=f"{col}=%{{y:.4g}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=df[time_col], y=df[col].rolling(roll,center=True).mean(),
                                 mode="lines", name="Rolling avg",
                                 line=dict(color=pal[i%len(pal)], width=2.5),
                                 hovertemplate=f"avg=%{{y:.4g}}<extra></extra>"))
        fig.update_layout(title=f"{vm['label']} {icon}  |  mean = {fmean:.4g} {vm.get('unit','')}",
                          xaxis_title=time_col, yaxis_title=vm.get("unit",""),
                          height=280, margin=dict(t=40,b=10,l=10,r=10), showlegend=False)
        with gcols[i % len(gcols)]:
            st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Steady-state assessment")
    ssrows = []
    for col in fcols:
        vm = meta["variables"].get(col, {"label": col, "unit": ""})
        is_ss, cv = _is_steady(df[col])
        sl = df[col].iloc[max(0, int(len(df)*0.8)):]
        status = "Steady ✅" if is_ss else ("Nearly steady ⚠️" if cv < 0.10 else "Transient ❌")
        ssrows.append({"Variable": vm["label"], "Final mean (last 20%)": sl.mean(),
                       "Std dev": sl.std(), "CV (std/mean)": cv, "Steady state?": status})
    def _css(v):
        if "Steady ✅" == v: return "background-color:#dcfce7;color:#15803d;font-weight:600"
        if "Nearly"    in v: return "background-color:#fef9c3;color:#92400e;font-weight:600"
        return "background-color:#fee2e2;color:#991b1b;font-weight:600"
    ssdf = pd.DataFrame(ssrows)
    st.dataframe(ssdf.style.format({"Final mean (last 20%)":"{:.4e}","Std dev":"{:.3e}","CV (std/mean)":"{:.4f}"})
                 .applymap(_css, subset=["Steady state?"]), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Physical interpretation")
    for col in fcols:
        vm = meta["variables"].get(col, {"label": col, "unit": "", "meaning": ""})
        is_ss, cv = _is_steady(df[col])
        fmean = df[col].iloc[max(0, int(len(df)*0.8)):].mean()
        with st.expander(f"**{vm['label']}** — mean = {fmean:.4e} {vm.get('unit','')}"):
            st.markdown(vm.get("meaning",""))
            if probe_type == "outlet" and col == "Ux" and fmean < 0:
                st.error("**Backflow detected!** Outlet Ux is negative. Extend the outlet domain or apply a backflow limiter.")
            elif probe_type == "wall" and col.startswith("U"):
                if abs(fmean) > 0.01:
                    st.warning(f"**No-slip check failed**: {col} = {fmean:.3e} m/s — expected ≈0. Check probe placement and wall BC.")
                else:
                    st.success(f"No-slip satisfied: {col} ≈ {fmean:.3e} m/s.")
            if not is_ss:
                st.warning(f"Not at steady state — CV = {cv:.3%}. Continue until CV < 2%.")
            else:
                st.success(f"Steady state reached — CV = {cv:.3%}.")


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Simulation/Mesh AI Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] { gap:6px; padding:4px; border-radius:10px; background:rgba(100,116,139,0.08); }
.stTabs [data-baseweb="tab"]      { padding:8px 20px; border-radius:8px; font-weight:500; font-size:0.9rem; }
[data-testid="stMetricValue"] { font-size:1.85rem !important; font-weight:700; }
[data-testid="stMetricLabel"] { font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.05em; color:#64748b; }
.status-good { color:#15803d; background:#dcfce7; padding:2px 10px; border-radius:20px; font-weight:700; font-size:.78rem; }
.status-warn { color:#92400e; background:#fef3c7; padding:2px 10px; border-radius:20px; font-weight:700; font-size:.78rem; }
.status-bad  { color:#991b1b; background:#fee2e2; padding:2px 10px; border-radius:20px; font-weight:700; font-size:.78rem; }
.streamlit-expanderHeader { font-weight:600; }
hr { margin:1.4rem 0 !important; border-color:#e2e8f0 !important; }
.stDownloadButton button, .stButton button { border-radius:8px; font-weight:500; }
[data-testid="stSidebar"] { min-width:260px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────

if "df" not in st.session_state:
    st.session_state.df = pd.read_csv(CSV_PATH) if CSV_PATH.exists() else pd.DataFrame()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🌊 SimScale AI Analysis")
    st.divider()

    df_sb = st.session_state.df
    st.markdown("**Dataset status**")
    if df_sb.empty:
        st.caption("No runs loaded yet.")
    else:
        labeled_n = int(df_sb["label"].notna().sum()) if "label" in df_sb.columns else 0
        st.caption(f"📋 {len(df_sb)} runs · {labeled_n} labeled · {len(df_sb)-labeled_n} unlabeled")
        if "overall_quality" in df_sb.columns:
            st.caption(f"🏆 Best: `{df_sb['overall_quality'].max():.3f}` · Worst: `{df_sb['overall_quality'].min():.3f}`")

    st.divider()

    with st.expander("🔢 y⁺ First-cell Calculator"):
        st.caption("Estimate first inflation layer height for a target y⁺.")
        sb_Re = st.number_input("Reynolds number",          value=1_000_000.0, format="%.2e", step=1e5, key="sb_re")
        sb_U  = st.number_input("Free-stream velocity (m/s)", value=30.0, key="sb_u")
        sb_nu = st.number_input("Kinematic viscosity ν (m²/s)", value=1.5e-5, format="%.2e", key="sb_nu")
        sb_yp = st.number_input("Target y⁺",                value=1.0, min_value=0.1, max_value=300.0, key="sb_yp")
        Cf    = 0.026 / (sb_Re ** (1/7))
        u_tau = np.sqrt(0.5 * Cf * sb_U**2)
        y1    = sb_yp * sb_nu / u_tau
        st.success(f"**y₁ ≈ {y1*1000:.4f} mm** ({y1:.3e} m)")
        st.caption(f"Cf ≈ {Cf:.3e} · u_τ ≈ {u_tau:.3f} m/s")
        if sb_yp <= 1:
            st.info("y⁺ ≤ 1 → Low-Re model (k-ω SST with low-Re correction).")
        elif sb_yp <= 5:
            st.warning("1 < y⁺ ≤ 5 → buffer layer. Avoid; target y⁺ < 1 or y⁺ > 30.")
        else:
            st.info("y⁺ > 30 → wall-function regime (standard k-ε or k-ω SST).")

    with st.expander("📐 Mesh Quality Reference"):
        ref = pd.DataFrame({
            "Metric":   ["nonOrtho avg", "nonOrtho max", "Skewness avg", "Skewness max", "AR avg", "Vol ratio max", "Overall score"],
            "Good":     ["< 30°",         "< 70°",        "< 0.5",         "< 2.0",        "< 3",    "< 10",          "> 0.75"],
            "Marginal": ["< 45°",         "< 85°",        "< 1.0",         "< 4.0",        "< 10",   "< 50",          "> 0.65"],
        })
        st.dataframe(ref, hide_index=True, width="stretch")

    st.divider()
    st.caption("Built for SimScale CFD workflows")

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("🌊 Simulation/Mesh AI Analysis")
st.caption("Parse SimScale mesh logs · Explore quality metrics · Train classifiers · Force & probe analysis")

tab_parse, tab_explore, tab_train, tab_forces, tab_probes = st.tabs([
    "📂  Parse Logs", "📊  Explore Data", "🤖  Train Models",
    "📈  Force Analysis", "🔬  Probe Analysis",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PARSE LOGS
# ════════════════════════════════════════════════════════════════════════════════

with tab_parse:
    col_up, col_prev = st.columns([1, 2], gap="large")

    with col_up:
        st.subheader("Upload mesh logs")
        uploaded_files = st.file_uploader(
            "Drop SimScale mesh log `.txt` files here",
            type="txt", accept_multiple_files=True,
            help="Copy log text from the SimScale UI → save as .txt → upload here.",
        )

    parsed_items: list[tuple[str, dict]] = []
    for uf in uploaded_files or []:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="wb") as tmp:
            tmp.write(uf.read())
            tmp_path = Path(tmp.name)
        row = parse_log_file(tmp_path)
        row["run_id"] = Path(uf.name).stem
        os.unlink(tmp_path)
        parsed_items.append((Path(uf.name).stem, row))

    with col_prev:
        if parsed_items:
            st.subheader("Parsed preview")
            preview_rows = []
            for stem, row in parsed_items:
                oq = row.get("overall_quality", float("nan"))
                lbl, cls = _quality_label(oq, "overall_quality") if not np.isnan(oq) else ("—", "")
                preview_rows.append({
                    "Run":             stem,
                    "Overall quality": round(oq, 4),
                    "Rating":          lbl,
                    "NonOrtho avg":    round(row.get("nonOrthogonality_average", float("nan")), 2),
                    "Skewness avg":    round(row.get("skewness_average", float("nan")), 4),
                    "AR avg":          round(row.get("aspectRatio_average", float("nan")), 4),
                    "Features":        len(row) - 1,
                })
            st.dataframe(pd.DataFrame(preview_rows), width="stretch", hide_index=True)

            # Key metrics vs thresholds
            st.divider()
            st.markdown("**Threshold check** — key metrics vs SimScale standards")
            for stem, row in parsed_items:
                st.markdown(f"*{stem}*")
                cols_th = st.columns(len(HIGHLIGHT_METRICS))
                for ci, metric in enumerate(HIGHLIGHT_METRICS):
                    val = row.get(metric)
                    if val is not None and not np.isnan(float(val)):
                        lbl, css = _quality_label(float(val), metric)
                        short = metric.replace("nonOrthogonality_","nonOrtho·").replace("skewness_","skew·").replace("aspectRatio_","AR·").replace("tetAspectRatio_","tetAR·").replace("volumeRatio_","volR·").replace("overall_","OQ·")
                        cols_th[ci].markdown(f"**{short}**  \n{float(val):.3g}  \n<span class='{css}'>{lbl}</span>", unsafe_allow_html=True)
        else:
            st.info("Upload log files on the left — parsed metrics will appear here.")

    if parsed_items:
        st.divider()
        st.subheader("Enter mesh parameters & assign quality labels")
        st.caption("Fill in the settings you used, assign a label based on Cl/Cd convergence, then click **Add to Dataset**.")
        with st.form("metadata_form"):
            meta_list = []
            for stem, row in parsed_items:
                oq = row.get("overall_quality", float("nan"))
                with st.expander(f"**{stem}**  ·  overall quality: `{oq:.4f}`", expanded=True):
                    c1, c2, c3, c4 = st.columns(4)
                    fineness  = c1.number_input("Mesh fineness (1–9)", 1, 9, 5,        key=f"{stem}_fn")
                    layers    = c1.number_input("Inflation layers",    0, 20, 3,        key=f"{stem}_il")
                    thickness = c2.number_input("1st layer (m)", value=0.001, format="%.5f", key=f"{stem}_ft")
                    growth    = c2.number_input("Growth rate",    value=1.2,   format="%.2f",  key=f"{stem}_gr")
                    cells     = c3.number_input("Cell count",     value=250_000, step=10_000,  key=f"{stem}_cc")
                    cl        = c3.number_input("Final Cl", value=0.0, format="%.4f",   key=f"{stem}_cl")
                    cd        = c4.number_input("Final Cd", value=0.0, format="%.5f",   key=f"{stem}_cd")
                    converged = c4.checkbox("Converged?", value=True,                   key=f"{stem}_cv")
                    label     = st.radio("Quality label", ["good","marginal","bad"],
                                         horizontal=True, key=f"{stem}_lbl",
                                         help="good = Cl/Cd converged · marginal = borderline · bad = poor")
                    meta_list.append({"stem":stem,"row":row,"meta":{
                        "mesh_fineness":fineness,"num_inflation_layers":layers,
                        "first_layer_thickness":thickness,"growth_rate":growth,
                        "cell_count":cells,"final_Cl":cl,"final_Cd":cd,
                        "converged":converged,"label":label}})
            submitted = st.form_submit_button("➕  Add to Dataset", type="primary")

        if submitted:
            df_cur = st.session_state.df
            for item in meta_list:
                merged = {**item["meta"], **item["row"], "run_id": item["stem"]}
                new_row = pd.DataFrame([merged])
                if not df_cur.empty and item["stem"] in df_cur["run_id"].values:
                    df_cur = df_cur[df_cur["run_id"] != item["stem"]]
                df_cur = pd.concat([df_cur, new_row], ignore_index=True)
            st.session_state.df = df_cur
            df_cur.to_csv(CSV_PATH, index=False)
            st.success(f"✅  Saved {len(meta_list)} run(s). Switch to **Explore Data** or **Train Models**.")
            st.rerun()

    st.divider()
    df = st.session_state.df
    if df.empty:
        st.info("No data yet — upload logs above to get started.")
    else:
        st.subheader(f"Current dataset  ·  {len(df)} run(s)")
        labeled_n = int(df["label"].notna().sum()) if "label" in df.columns else 0
        feat_n    = len([c for c in df.columns if c not in NON_FEATURE_COLS and c != "run_id"])
        m1,m2,m3  = st.columns(3)
        m1.metric("Total runs",      len(df))
        m2.metric("Labeled",         labeled_n)
        m3.metric("Feature columns", feat_n)
        dcols = [c for c in ["run_id","label","overall_quality","mesh_fineness",
                              "num_inflation_layers","cell_count","nonOrthogonality_average","skewness_average"] if c in df.columns]
        st.dataframe(df[dcols], width="stretch", hide_index=True)
        st.download_button("⬇️  Download full CSV", df.to_csv(index=False).encode(),
                           file_name="mesh_dataset.csv", mime="text/csv")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORE DATA
# ════════════════════════════════════════════════════════════════════════════════

with tab_explore:
    df = st.session_state.df
    if df.empty:
        st.info("No data yet — go to **Parse Logs** to add runs.")
    else:
        labeled_n = int(df["label"].notna().sum()) if "label" in df.columns else 0
        feat_n    = len([c for c in df.columns if c not in NON_FEATURE_COLS and c != "run_id"])
        m1,m2,m3  = st.columns(3)
        m1.metric("Runs", len(df))
        m2.metric("Labeled", labeled_n)
        m3.metric("Features", feat_n)

        # ── Overall quality bar ───────────────────────────────────────────────
        if "overall_quality" in df.columns:
            st.subheader("Overall mesh quality score")
            chart_df = df[["run_id","overall_quality"]].copy()
            chart_df["label"] = df["label"].fillna("unlabeled") if "label" in df.columns else "unlabeled"
            fig_bar = px.bar(chart_df.sort_values("overall_quality", ascending=False),
                             x="run_id", y="overall_quality", color="label",
                             color_discrete_map=LABEL_COLORS, category_orders={"label":LABEL_ORDER},
                             height=340, text_auto=".3f",
                             labels={"run_id":"Run","overall_quality":"Quality Score"})
            fig_bar.add_hline(y=0.75, line_dash="dash", line_color="#22c55e", annotation_text="0.75 good",      annotation_position="top right")
            fig_bar.add_hline(y=0.65, line_dash="dot",  line_color="#f59e0b", annotation_text="0.65 marginal",  annotation_position="bottom right")
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(yaxis_range=[0,1.15], yaxis_title="Score (0–1)", xaxis_title=None, margin=dict(t=20,b=10))
            st.plotly_chart(fig_bar, width="stretch")

        # ── Key metrics heatmap ───────────────────────────────────────────────
        st.divider()
        st.subheader("Key metrics — per run heatmap")
        avail = [m for m in HIGHLIGHT_METRICS if m in df.columns]
        if avail:
            tbl = df[["run_id"] + (["label"] if "label" in df.columns else []) + avail].copy()
            short = lambda c: (c.replace("nonOrthogonality_","nonOrtho·")
                                .replace("skewness_","skew·").replace("aspectRatio_","AR·")
                                .replace("tetAspectRatio_","tetAR·").replace("volumeRatio_","volR·"))
            tbl.columns = [short(c) for c in tbl.columns]
            num_c = [c for c in tbl.columns if c not in ("run_id","label")]
            st.dataframe(
                tbl.set_index("run_id").style
                   .background_gradient(subset=num_c, cmap="RdYlGn_r", axis=0)
                   .format({c:"{:.4f}" for c in num_c}),
                width="stretch")

        # ── Box plots by label ────────────────────────────────────────────────
        if labeled_n >= 2:
            st.divider()
            st.subheader("Metric distribution by quality label")
            st.caption("Box plots show how each metric separates across good / marginal / bad runs.")
            box_metrics = [m for m in HIGHLIGHT_METRICS if m in df.columns]
            sel_box = st.selectbox("Select metric", box_metrics,
                                   index=box_metrics.index("overall_quality") if "overall_quality" in box_metrics else 0,
                                   key="box_sel")
            box_df = df.dropna(subset=["label"])
            fig_box = px.box(box_df, x="label", y=sel_box, color="label",
                             color_discrete_map=LABEL_COLORS, category_orders={"label":LABEL_ORDER},
                             points="all", height=360,
                             labels={"label":"Quality label", sel_box: sel_box})
            t_info = MESH_THRESHOLDS.get(sel_box, {})
            if t_info:
                for lvl, col, pos in [(t_info["good"],"#22c55e","top right"),(t_info["marginal"],"#f59e0b","bottom right")]:
                    fig_box.add_hline(y=lvl, line_dash="dot", line_color=col,
                                      annotation_text=f"{'good' if col=='#22c55e' else 'marginal'} threshold",
                                      annotation_position=pos)
            fig_box.update_layout(margin=dict(t=20,b=10), showlegend=False)
            st.plotly_chart(fig_box, width="stretch")

        # ── Scatter comparison ────────────────────────────────────────────────
        if len(df) >= 2:
            st.divider()
            st.subheader("Scatter comparison")
            num_cols_all = sorted([c for c in df.columns if c not in NON_FEATURE_COLS
                                   and c != "run_id" and pd.api.types.is_numeric_dtype(df[c])])
            c1, c2 = st.columns(2)
            xi = num_cols_all.index("nonOrthogonality_average") if "nonOrthogonality_average" in num_cols_all else 0
            yi = num_cols_all.index("overall_quality") if "overall_quality" in num_cols_all else min(1, len(num_cols_all)-1)
            xax = c1.selectbox("X axis", num_cols_all, index=xi, key="sc_x")
            yax = c2.selectbox("Y axis", num_cols_all, index=yi, key="sc_y")
            color_by = "label" if "label" in df.columns and df["label"].notna().any() else None
            sc_df = df.dropna(subset=["label"]) if color_by else df
            fig_sc = px.scatter(sc_df, x=xax, y=yax, color=color_by,
                                color_discrete_map=LABEL_COLORS if color_by else None,
                                category_orders={"label":LABEL_ORDER},
                                text="run_id", height=360,
                                labels={xax:xax, yax:yax})
            fig_sc.update_traces(textposition="top center", marker=dict(size=10))
            fig_sc.update_layout(margin=dict(t=20,b=10))
            st.plotly_chart(fig_sc, width="stretch")

        # ── Correlation heatmap ───────────────────────────────────────────────
        if len(df) >= 3:
            st.divider()
            st.subheader("Feature correlation matrix")
            st.caption("Pearson correlation among key mesh metrics. Helps identify redundant features.")
            corr_cols = [m for m in HIGHLIGHT_METRICS if m in df.columns]
            if len(corr_cols) >= 3:
                corr_m = df[corr_cols].corr()
                short2 = lambda c: c.replace("nonOrthogonality_","nO·").replace("skewness_","sk·").replace("aspectRatio_","AR·").replace("tetAspectRatio_","tAR·").replace("volumeRatio_","vR·").replace("overall_quality","OQ")
                corr_m.columns = [short2(c) for c in corr_m.columns]
                corr_m.index   = corr_m.columns
                fig_corr = px.imshow(corr_m, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                     text_auto=".2f", height=420,
                                     labels=dict(color="Pearson r"))
                fig_corr.update_layout(margin=dict(t=20,b=10))
                st.plotly_chart(fig_corr, width="stretch")

        # ── Radar chart ───────────────────────────────────────────────────────
        st.divider()
        st.subheader("Metric profile — radar chart")
        st.caption("Normalized to [0,1] within dataset. Lower = better for most axes.")
        rcols = [m for m in RADAR_METRICS if m in df.columns]
        if len(rcols) >= 4:
            rdf = df[["run_id"] + rcols].copy()
            for col in rcols:
                lo, hi = rdf[col].min(), rdf[col].max()
                rdf[col] = (rdf[col]-lo)/(hi-lo) if hi != lo else 0.5
            theta = [c.replace("nonOrthogonality_","nO·").replace("skewness_","sk·")
                      .replace("aspectRatio_","AR·").replace("tetAspectRatio_","tAR·")
                      .replace("volumeRatio_","vR·").replace("tetEdgeRatio_","tER·") for c in rcols]
            pal = px.colors.qualitative.Set2
            fig_rad = go.Figure()
            for i, (_, rr) in enumerate(rdf.iterrows()):
                v = [rr[c] for c in rcols]
                fig_rad.add_trace(go.Scatterpolar(r=v+[v[0]], theta=theta+[theta[0]],
                                                   fill="toself", name=rr["run_id"],
                                                   line_color=pal[i%len(pal)], opacity=0.75))
            fig_rad.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),
                                  height=440, margin=dict(t=20,b=10))
            st.plotly_chart(fig_rad, width="stretch")

        # ── Feature distribution ──────────────────────────────────────────────
        if len(df) >= 2:
            st.divider()
            st.subheader("Feature distribution")
            numeric_cols = sorted([c for c in df.columns if c not in NON_FEATURE_COLS
                                   and c != "run_id" and pd.api.types.is_numeric_dtype(df[c])])
            def_i = numeric_cols.index("overall_quality") if "overall_quality" in numeric_cols else 0
            sel   = st.selectbox("Select a feature", numeric_cols, index=def_i, key="hist_sel")
            col_b = "label" if "label" in df.columns and df["label"].notna().any() else None
            fig_h = px.histogram(df if not col_b else df.dropna(subset=["label"]),
                                 x=sel, color=col_b, color_discrete_map=LABEL_COLORS if col_b else None,
                                 barmode="overlay", nbins=25, opacity=0.75, height=280)
            fig_h.update_layout(margin=dict(t=10,b=10), xaxis_title=sel)
            st.plotly_chart(fig_h, width="stretch")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRAIN MODELS
# ════════════════════════════════════════════════════════════════════════════════

with tab_train:
    df = st.session_state.df
    if df.empty or "label" not in df.columns:
        st.info("No labeled data yet — go to **Parse Logs** and add runs with labels.")
    else:
        labeled_df = df.dropna(subset=["label"])
        if len(labeled_df) < 6:
            st.warning(f"Only **{len(labeled_df)}** labeled run(s). Add **{6-len(labeled_df)}** more for cross-validation. Aim for 15–30 runs.")
        else:
            fcols = [c for c in labeled_df.columns if c not in NON_FEATURE_COLS
                     and c != "run_id" and pd.api.types.is_numeric_dtype(labeled_df[c])
                     and labeled_df[c].notna().any()]
            X = labeled_df[fcols].fillna(labeled_df[fcols].median())
            y = labeled_df["label"]

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Labeled runs", len(labeled_df))
            m2.metric("Features",     len(fcols))
            m3.metric("Classes",      y.nunique())
            m4.metric("CV folds",     min(5, len(labeled_df)))

            st.subheader("Class distribution")
            dist_df = y.value_counts().reset_index(); dist_df.columns = ["label","count"]
            fig_dist = px.bar(dist_df, x="label", y="count", color="label",
                              color_discrete_map=LABEL_COLORS, text_auto=True, height=220)
            fig_dist.update_layout(showlegend=False, margin=dict(t=10,b=10), xaxis_title=None)
            fig_dist.update_traces(textposition="outside")
            st.plotly_chart(fig_dist, width="stretch")
            st.divider()

            if st.button("🚀  Train & evaluate all models", type="primary"):
                cv_k = min(5, len(labeled_df))
                models = {
                    "Gaussian Naive Bayes": Pipeline([("sc",StandardScaler()),("nb",GaussianNB())]),
                    "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42),
                    "Logistic Regression":  Pipeline([("sc",StandardScaler()),("lr",LogisticRegression(max_iter=2000))]),
                }
                results = []
                with st.spinner("Training…"):
                    for name, model in models.items():
                        scores = cross_val_score(model, X, y, cv=cv_k, scoring="accuracy")
                        results.append({"Model":name,"CV Accuracy":float(scores.mean()),"± Std":float(scores.std())})
                rdf2 = pd.DataFrame(results).sort_values("CV Accuracy", ascending=False)

                st.subheader("Model comparison")
                fig_m = go.Figure(go.Bar(
                    x=rdf2["Model"], y=rdf2["CV Accuracy"],
                    error_y=dict(type="data", array=rdf2["± Std"].tolist()),
                    marker_color=["#6366f1","#22c55e","#f59e0b"][:len(rdf2)],
                    text=[f"{v:.3f}" for v in rdf2["CV Accuracy"]], textposition="outside"))
                fig_m.update_layout(yaxis_range=[0,1.15], yaxis_title="CV Accuracy",
                                    xaxis_title=None, height=300, margin=dict(t=10,b=10))
                st.plotly_chart(fig_m, width="stretch")

                # Confusion matrix
                st.divider()
                best_name  = rdf2.iloc[0]["Model"]
                best_model = models[best_name]
                strat = y if y.value_counts().min() >= 2 else None
                Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=strat)
                best_model.fit(Xtr, ytr)
                ypred = best_model.predict(Xte)
                labs  = sorted(y.unique())
                cm    = confusion_matrix(yte, ypred, labels=labs)
                st.subheader(f"Confusion matrix — {best_name}")
                fig_cm = px.imshow(cm, x=labs, y=labs, color_continuous_scale="Blues",
                                   text_auto=True, labels=dict(x="Predicted",y="True",color="Count"), height=340)
                fig_cm.update_layout(margin=dict(t=20,b=10))
                st.plotly_chart(fig_cm, width="stretch")

                # Feature importance
                st.divider()
                st.subheader("Top 15 predictive features — Random Forest")
                rf = RandomForestClassifier(n_estimators=200, random_state=42)
                rf.fit(X, y)
                imp_df = (pd.DataFrame({"feature":fcols,"importance":rf.feature_importances_})
                          .sort_values("importance", ascending=False).head(15))
                fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h",
                                 color="importance", color_continuous_scale="Blues", height=460,
                                 labels={"importance":"Importance","feature":"Feature"})
                fig_imp.update_layout(yaxis=dict(autorange="reversed"), showlegend=False,
                                      coloraxis_showscale=False, margin=dict(t=10,b=10))
                st.plotly_chart(fig_imp, width="stretch")

                # Predict unlabeled runs
                unlabeled_df = df[df["label"].isna()] if "label" in df.columns else pd.DataFrame()
                if not unlabeled_df.empty:
                    st.divider()
                    st.subheader(f"Predictions for {len(unlabeled_df)} unlabeled run(s)")
                    st.caption(f"Using the best model: **{best_name}**")
                    X_ul = unlabeled_df[fcols].fillna(labeled_df[fcols].median())
                    best_model.fit(X, y)
                    preds = best_model.predict(X_ul)
                    pred_df = unlabeled_df[["run_id"] + [c for c in ["overall_quality","nonOrthogonality_average","skewness_average"] if c in unlabeled_df.columns]].copy()
                    pred_df["Predicted label"] = preds
                    def _pc(v):
                        return {"good":"background-color:#dcfce7","marginal":"background-color:#fef9c3","bad":"background-color:#fee2e2"}.get(v,"")
                    st.dataframe(pred_df.style.applymap(_pc, subset=["Predicted label"]),
                                 width="stretch", hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — FORCE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

with tab_forces:
    st.subheader("Force & moment coefficient analysis")
    st.caption("Upload the force-coefficient CSV from SimScale. Each coefficient is plotted individually for clarity.")

    force_file = st.file_uploader("Upload force-coefficient CSV", type=["csv"], key="force_csv",
                                   help="Expected columns: time column + FORCE_COEFFICIENT_* columns.")

    if force_file is not None:
        try:
            fdf = pd.read_csv(force_file)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            fdf = None

        if fdf is not None:
            fdf.columns = fdf.columns.str.strip()
            tcands = [c for c in fdf.columns if "time" in c.lower() or "iteration" in c.lower()]
            time_col = tcands[0] if tcands else fdf.columns[0]
            coeff_cols = [c for c in fdf.columns if c != time_col and pd.api.types.is_numeric_dtype(fdf[c])]

            if not coeff_cols:
                st.error("No numeric coefficient columns found.")
            else:
                fdf = fdf[[time_col]+coeff_cols].dropna()

                c1, c2 = st.columns([2,1])
                selected_cols = c1.multiselect("Coefficients to display", coeff_cols, default=coeff_cols)
                window_pct    = c2.slider("Convergence window (%)", 5, 50, 20, 5,
                                          help="Stats computed over this trailing % of the run.")
                if not selected_cols:
                    selected_cols = coeff_cols
                n_window  = max(2, int(len(fdf)*window_pct/100))
                window_df = fdf.tail(n_window)

                # ── Individual subplots per coefficient ───────────────────────
                st.divider()
                st.subheader("Coefficient history — individual plots")
                roll_n = max(3, len(fdf)//10)
                pal    = px.colors.qualitative.Plotly
                n_cols_grid = min(2, len(selected_cols))
                gcols = st.columns(n_cols_grid)
                x_start = float(fdf[time_col].iloc[-n_window])
                x_end   = float(fdf[time_col].iloc[-1])

                for i, col in enumerate(selected_cols):
                    rolled = fdf[col].rolling(roll_n, center=True).mean()
                    fig_c  = go.Figure()
                    fig_c.add_trace(go.Scatter(x=fdf[time_col], y=fdf[col], mode="lines",
                                               name="Raw", line=dict(color=pal[i%len(pal)], width=1.2),
                                               opacity=0.45, hovertemplate=f"{col}=%{{y:.5g}}<extra></extra>"))
                    fig_c.add_trace(go.Scatter(x=fdf[time_col], y=rolled, mode="lines",
                                               name="Rolling avg", line=dict(color=pal[i%len(pal)], width=2.2),
                                               hovertemplate=f"avg=%{{y:.5g}}<extra></extra>"))
                    fig_c.add_vrect(x0=x_start, x1=x_end, fillcolor="#6366f1", opacity=0.07, line_width=0)
                    wm = window_df[col].mean()
                    fig_c.add_hline(y=wm, line_dash="dot", line_color="#64748b",
                                    annotation_text=f"mean={wm:.4g}", annotation_position="top left")
                    fig_c.update_layout(title=col, xaxis_title=time_col, height=280,
                                        margin=dict(t=35,b=10,l=10,r=10), showlegend=False)
                    with gcols[i % n_cols_grid]:
                        st.plotly_chart(fig_c, width="stretch")

                # ── Convergence table ─────────────────────────────────────────
                st.divider()
                st.subheader("Convergence assessment")
                st.caption(f"Statistics over last **{window_pct}%** ({n_window} steps).")

                rows = []
                for col in selected_cols:
                    wv = window_df[col]
                    mean_val = wv.mean(); std_val = wv.std()
                    cv = abs(std_val/mean_val) if mean_val != 0 else float("inf")
                    total_chg = abs(fdf[col].iloc[-1]-fdf[col].iloc[0])
                    status, _ = convergence_status(cv)
                    rows.append({"Coefficient":col, "Min":fdf[col].min(), "Max":fdf[col].max(),
                                 "Final mean":mean_val, "Std dev":std_val,
                                 "CV (std/mean)":cv, "Total change":total_chg, "Status":status})

                conv_df = pd.DataFrame(rows)
                def _cs(v):
                    if "Converged ✅" == v: return "background-color:#dcfce7;color:#15803d;font-weight:600"
                    if "Marginal"    in v:  return "background-color:#fef9c3;color:#92400e;font-weight:600"
                    return "background-color:#fee2e2;color:#991b1b;font-weight:600"
                fmt = {"Min":"{:.4e}","Max":"{:.4e}","Final mean":"{:.4e}","Std dev":"{:.4e}",
                       "CV (std/mean)":"{:.4f}","Total change":"{:.4e}"}
                st.dataframe(conv_df.style.format(fmt).applymap(_cs, subset=["Status"]),
                             width="stretch", hide_index=True)

                n_conv = sum(1 for r in rows if "Converged ✅" == r["Status"])
                n_marg = sum(1 for r in rows if "Marginal"    in r["Status"])
                n_bad  = sum(1 for r in rows if "Not converged" in r["Status"])
                final_means = {r["Coefficient"]: r["Final mean"] for r in rows}

                # ── Derived aerodynamic metrics ───────────────────────────────
                st.divider()
                st.subheader("Aerodynamic metrics")
                def coeff_suffix(col): return col.split("_")[-1].upper()
                cd_col  = next((c for c in selected_cols if coeff_suffix(c)=="CD"),  None)
                cl_col  = next((c for c in selected_cols if coeff_suffix(c)=="CL"),  None)
                clf_col = next((c for c in selected_cols if coeff_suffix(c)=="CLF"), None)
                clr_col = next((c for c in selected_cols if coeff_suffix(c)=="CLR"), None)
                cm_col  = next((c for c in selected_cols if coeff_suffix(c)=="CM"),  None)

                metric_cards = []
                if cd_col:
                    metric_cards.append(("Cd (final mean)", f"{final_means[cd_col]:.4e}", "Drag coefficient. Lower = less drag."))
                if cl_col:
                    clv = final_means[cl_col]
                    metric_cards.append(("Cl (final mean)", f"{clv:.4e}", f"{'Downforce ✅ (negative)' if clv<0 else 'Lift (positive)'}. For ground vehicles, negative Cl is desirable."))
                if cd_col and cl_col and final_means[cd_col] != 0:
                    ld = abs(final_means[cl_col]/final_means[cd_col])
                    metric_cards.append(("L/D ratio", f"{ld:.2f}", "Aerodynamic efficiency: Cl÷Cd. Higher = more force per unit drag."))
                if clf_col and clr_col:
                    tot = final_means[clf_col]+final_means[clr_col]
                    if tot != 0:
                        bal = final_means[clf_col]/tot*100
                        metric_cards.append(("Aero balance", f"{bal:.1f}% front", "Front÷(Front+Rear) downforce. 42–58% front = neutral balance."))
                if cm_col:
                    metric_cards.append(("Cm (final mean)", f"{final_means[cm_col]:.4e}", "Pitching moment. Near 0 = balanced pitch forces."))
                if metric_cards:
                    mc = st.columns(len(metric_cards))
                    for i,(lbl,val,hlp) in enumerate(metric_cards):
                        mc[i].metric(lbl, val, help=hlp)

                # ── Trend analysis ────────────────────────────────────────────
                st.divider()
                st.subheader("Trend analysis")
                st.caption(f"Linear drift and oscillation flag over the convergence window (last {window_pct}%, {n_window} steps).")
                trend_rows = []
                for col in selected_cols:
                    xw = window_df[time_col].values.astype(float)
                    yw = window_df[col].values.astype(float)
                    slope = float(np.polyfit(xw-xw.mean(), yw, 1)[0])
                    ma = abs(final_means[col])
                    drift_pct = abs(slope)/ma*100 if ma > 0 else float("inf")
                    diffs = np.diff(yw)
                    osc   = int(np.sum(np.diff(np.sign(diffs))!=0)) > len(diffs)*0.4
                    dir_  = "Rising ↑" if slope>1e-12 else ("Falling ↓" if slope<-1e-12 else "Flat →")
                    trend_rows.append({"Coefficient":col, "Slope (per unit)":slope,
                                       "Drift %/100":drift_pct, "Trend":dir_,
                                       "Oscillating?":"Yes ↕" if osc else "No →"})
                trend_df = pd.DataFrame(trend_rows)
                st.dataframe(trend_df.style.format({"Slope (per unit)":"{:.4e}","Drift %/100":"{:.2f}"}),
                             width="stretch", hide_index=True)

                # ── CFD insights ──────────────────────────────────────────────
                st.divider()
                st.subheader("What the data is telling you")
                insights = []
                if n_bad==0 and n_marg==0:
                    insights.append(("success", f"All {len(selected_cols)} coefficient(s) **converged** (CV < 1% in last {window_pct}%). Safe to use final mean values."))
                elif n_bad==0:
                    insights.append(("warning", f"{n_conv} converged, {n_marg} **marginal**. Run ~{int(len(fdf)*0.5)} more steps and recheck."))
                else:
                    insights.append(("error", f"{n_bad} coefficient(s) **still drifting** (CV ≥ 5%). Do not extract values yet — extend the run."))
                fast = [r for r in trend_rows if r["Drift %/100"]>5]
                if fast:
                    insights.append(("error", f"**Fast drift** in {', '.join(r['Coefficient'] for r in fast)}: >5%/100 time units. Solver has not settled."))
                ld_card = next((c for c in metric_cards if c[0]=="L/D ratio"), None)
                if ld_card:
                    lv = float(ld_card[1])
                    if lv>10:   insights.append(("success", f"**L/D = {lv:.1f}** — excellent aerodynamic efficiency."))
                    elif lv>3:  insights.append(("info",    f"**L/D = {lv:.1f}** — moderate. Consider drag reduction (wake smoothing, frontal area)."))
                    else:       insights.append(("warning", f"**L/D = {lv:.1f}** — drag-dominated. Review bluff bodies and separation zones."))
                bal_card = next((c for c in metric_cards if c[0]=="Aero balance"), None)
                if bal_card:
                    bv = float(bal_card[1].split("%")[0])
                    if 42<=bv<=58: insights.append(("success", f"**Aero balance {bv:.1f}% front** — neutral. Even axle loading supports balanced handling."))
                    elif bv>58:    insights.append(("warning", f"**Aero balance {bv:.1f}% front** — front-heavy → understeer tendency. Reduce front wing/diffuser angle."))
                    else:          insights.append(("warning", f"**Aero balance {bv:.1f}% front** — rear-heavy → oversteer tendency. Reduce rear wing angle or increase front splitter."))
                if cm_col:
                    cmv = final_means[cm_col]
                    cdr = abs(final_means[cd_col]) if cd_col else 1
                    if abs(cmv)<cdr*0.5: insights.append(("info",    f"**Cm ≈ {cmv:.3e}** — small; balanced pitch forces."))
                    elif cmv>0:          insights.append(("warning", f"**Cm = {cmv:.3e} (nose-up)** — load transfers to rear at speed; check front-end geometry."))
                    else:                insights.append(("warning", f"**Cm = {cmv:.3e} (nose-down)** — increased front load; monitor for front-axle overloading."))
                osc_cols = [r["Coefficient"] for r in trend_rows if "Yes" in r["Oscillating?"]]
                if osc_cols:
                    insights.append(("warning", f"**Oscillating**: {', '.join(osc_cols)}. Possible causes: unsteady separation, vortex shedding, or coarse time step."))
                if cl_col and final_means[cl_col]>0:
                    insights.append(("warning", "**Cl > 0 (lift, not downforce)**. For ground vehicles this reduces grip. Verify configuration intent."))
                for sev, msg in insights:
                    getattr(st, sev)(msg)

                # Final values summary
                st.divider()
                with st.expander("Final values — mean ± std over convergence window", expanded=True):
                    for r in rows:
                        icon = "✅" if "Converged" in r["Status"] else ("⚠️" if "Marginal" in r["Status"] else "❌")
                        st.markdown(f"{icon} **{r['Coefficient']}**: `{r['Final mean']:.6e}` ± `{r['Std dev']:.2e}`")

                with st.expander("📖 Coefficient reference guide"):
                    st.markdown("""
| Coefficient | What it measures | Good direction | Typical range |
|---|---|---|---|
| **Cd** | Drag — resistance to forward motion | Lower | 0.1–0.5 (streamlined) |
| **Cl** | Total lift/downforce | Negative = downforce | −3 to +1 |
| **Clf** | Front-axle lift component | Negative = front downforce | — |
| **Clr** | Rear-axle lift component | Negative = rear downforce | — |
| **Cm** | Pitching moment | Near 0 = balanced | −0.5 to +0.5 |
| **L/D** | Aerodynamic efficiency | Higher | 1–20+ |
| **Aero balance** | Front ÷ (Front+Rear) downforce | 42–58% front = neutral | — |

**Convergence criteria (CV = std ÷ |mean| in convergence window):**
- ✅ Converged — CV < 1%   · ⚠️ Marginal — CV 1–5%   · ❌ Not converged — CV > 5%
""")
    else:
        st.info("Upload a force-coefficient CSV above to get started.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — PROBE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

with tab_probes:
    st.subheader("Probe & residual analysis")
    st.caption("Upload CSV files from SimScale: **Residuals**, **Domain**, **Inlets**, **Outlets**, **Walls**. "
               "File type is auto-detected from the filename.")

    probe_files = st.file_uploader("Upload probe CSV files (one or more)", type=["csv"],
                                    accept_multiple_files=True, key="probe_csvs",
                                    help="Any combination of: Residuals.csv, Domain.csv, Inlets.csv, Outlets.csv, Walls.csv")

    if probe_files:
        for probe_file in probe_files:
            st.divider()
            st.markdown(f"### 📄 {probe_file.name}")
            try:
                pdf = pd.read_csv(probe_file)
            except Exception as e:
                st.error(f"Could not parse {probe_file.name}: {e}")
                continue
            pdf.columns = pdf.columns.str.strip()
            tcands    = [c for c in pdf.columns if "time" in c.lower() or "iteration" in c.lower()]
            tcp       = tcands[0] if tcands else pdf.columns[0]
            ptype     = _detect_probe_type(probe_file.name, list(pdf.columns))
            ptype_map = {"residuals":"Residuals","inlet":"Inlet","outlet":"Outlet","wall":"Wall","domain":"Domain"}
            badge_cls = {"residuals":"status-warn","inlet":"status-good","outlet":"status-good","wall":"status-good","domain":"status-good"}
            st.markdown(
                f"<span class='{badge_cls.get(ptype,'status-good')}'>{ptype_map.get(ptype,ptype.title())}</span>"
                f"&nbsp; {len(pdf)} time steps · {len(pdf.columns)-1} variable(s) · time col: `{tcp}`",
                unsafe_allow_html=True)
            if ptype == "residuals":
                _render_residuals(pdf, tcp)
            else:
                _render_field_probe(pdf, tcp, ptype)
    else:
        st.info("Upload probe CSV files above. In SimScale, export from **Simulation Results → Result Control → Export CSV**.")
        with st.expander("What each file type contains"):
            st.markdown("""
| File | What it shows | Key variables |
|---|---|---|
| **Residuals.csv** | Solver equation residuals — must decrease to confirm convergence | Ux, Uy, Uz, p, k, ω, h |
| **Domain.csv** | Field values at a monitoring point in bulk flow | T, Ux, Uy, Uz, p |
| **Inlets.csv** | Values at inlet boundary — should match inlet BC | T, Ux, Uy, Uz, p |
| **Outlets.csv** | Values at outlet — backflow detection | T, Ux, Uy, Uz, p |
| **Walls.csv** | Values at wall surface — velocity must be ~0 (no-slip) | T, Ux, Uy, Uz, p |
""")
