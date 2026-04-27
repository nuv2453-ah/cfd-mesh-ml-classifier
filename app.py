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
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from simscale_log_parser import METRIC_NAMES, parse_log_file

# ── Constants ─────────────────────────────────────────────────────────────────

CSV_PATH = Path("mesh_dataset.csv")

NON_FEATURE_COLS = {"run_id", "label", "final_Cl", "final_Cd", "converged"}

LABEL_COLORS = {"good": "#22c55e", "marginal": "#f59e0b", "bad": "#ef4444", "unlabeled": "#94a3b8"}
LABEL_ORDER = ["good", "marginal", "bad", "unlabeled"]

HIGHLIGHT_METRICS = [
    "overall_quality",
    "nonOrthogonality_average",
    "nonOrthogonality_p99_9",
    "skewness_average",
    "skewness_max",
    "aspectRatio_average",
    "tetAspectRatio_max",
    "volumeRatio_max",
]

RADAR_METRICS = [
    "nonOrthogonality_average",
    "nonOrthogonality_p99_9",
    "skewness_average",
    "skewness_max",
    "aspectRatio_average",
    "tetAspectRatio_average",
    "volumeRatio_average",
    "tetEdgeRatio_average",
]

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Simulation/Mesh AI Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* Tighten tab bar padding */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 6px 6px 0 0; }

    /* Metric cards */
    [data-testid="stMetricValue"] { font-size: 2rem; }

    /* Subtle card background for expanders */
    .streamlit-expanderHeader { font-weight: 600; }

    /* Download button */
    .stDownloadButton button { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "df" not in st.session_state:
    st.session_state.df = pd.read_csv(CSV_PATH) if CSV_PATH.exists() else pd.DataFrame()

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🌊 Simulation/Mesh AI Analysis")
st.caption("Parse SimScale mesh logs · Explore quality metrics · Train & compare classifiers · Analyse force coefficients")

tab_parse, tab_explore, tab_train, tab_forces = st.tabs(
    ["📂  Parse Logs", "📊  Explore Data", "🤖  Train Models", "📈  Force Analysis"]
)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PARSE LOGS
# ════════════════════════════════════════════════════════════════════════════════

with tab_parse:
    col_up, col_prev = st.columns([1, 2], gap="large")

    with col_up:
        st.subheader("Upload logs")
        uploaded_files = st.file_uploader(
            "Drop SimScale mesh log `.txt` files here",
            type="txt",
            accept_multiple_files=True,
            help="Copy the log text from the SimScale UI → paste into a .txt file → upload here.",
        )

    # Parse every uploaded file
    parsed_items: list[tuple[str, dict]] = []
    for uf in uploaded_files or []:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="wb") as tmp:
            tmp.write(uf.read())
            tmp_path = Path(tmp.name)
        row = parse_log_file(tmp_path)
        stem = Path(uf.name).stem
        row["run_id"] = stem
        os.unlink(tmp_path)
        parsed_items.append((stem, row))

    with col_prev:
        if parsed_items:
            st.subheader("Parsed preview")
            preview_rows = []
            for stem, row in parsed_items:
                preview_rows.append({
                    "Run": stem,
                    "Overall quality": round(row.get("overall_quality", float("nan")), 4),
                    "NonOrtho avg": round(row.get("nonOrthogonality_average", float("nan")), 3),
                    "Skewness avg": round(row.get("skewness_average", float("nan")), 4),
                    "AR avg": round(row.get("aspectRatio_average", float("nan")), 4),
                    "Features": len(row) - 1,
                })
            st.dataframe(pd.DataFrame(preview_rows), width="stretch", hide_index=True)
        else:
            st.info("Upload log files on the left — parsed metrics will appear here.")

    # Metadata form
    if parsed_items:
        st.divider()
        st.subheader("Enter mesh parameters & assign quality labels")
        st.caption(
            "Fill in the simulation settings you used for each run, "
            "assign a label based on Cl/Cd convergence, then click **Add to Dataset**."
        )

        with st.form("metadata_form"):
            meta_list = []
            for stem, row in parsed_items:
                oq = row.get("overall_quality", float("nan"))
                with st.expander(f"**{stem}**  ·  overall quality score: `{oq:.4f}`", expanded=True):
                    c1, c2, c3, c4 = st.columns(4)
                    fineness   = c1.number_input("Mesh fineness (1–9)", 1, 9, 5, key=f"{stem}_fn")
                    layers     = c1.number_input("Inflation layers", 0, 20, 3, key=f"{stem}_il")
                    thickness  = c2.number_input("1st layer (m)", value=0.001, format="%.5f", key=f"{stem}_ft")
                    growth     = c2.number_input("Growth rate", value=1.2, format="%.2f", key=f"{stem}_gr")
                    cells      = c3.number_input("Cell count", value=250_000, step=10_000, key=f"{stem}_cc")
                    cl         = c3.number_input("Final Cl", value=0.0, format="%.4f", key=f"{stem}_cl")
                    cd         = c4.number_input("Final Cd", value=0.0, format="%.5f", key=f"{stem}_cd")
                    converged  = c4.checkbox("Converged?", value=True, key=f"{stem}_cv")
                    label = st.radio(
                        "Mesh quality label",
                        ["good", "marginal", "bad"],
                        horizontal=True,
                        key=f"{stem}_lbl",
                        help="good = Cl/Cd well converged · marginal = borderline · bad = poor convergence",
                    )
                    meta_list.append({
                        "stem": stem,
                        "row": row,
                        "meta": {
                            "mesh_fineness": fineness,
                            "num_inflation_layers": layers,
                            "first_layer_thickness": thickness,
                            "growth_rate": growth,
                            "cell_count": cells,
                            "final_Cl": cl,
                            "final_Cd": cd,
                            "converged": converged,
                            "label": label,
                        },
                    })

            submitted = st.form_submit_button(
                "➕  Add to Dataset", type="primary"
            )

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
            st.success(
                f"✅  Saved {len(meta_list)} run(s) to `{CSV_PATH}`.  "
                "Switch to **Explore Data** or **Train Models**."
            )
            st.rerun()

    # Current dataset summary
    st.divider()
    df = st.session_state.df
    if df.empty:
        st.info("No data yet — upload logs above to get started.")
    else:
        st.subheader(f"Current dataset  ·  {len(df)} run(s)")

        labeled_n = int(df["label"].notna().sum()) if "label" in df.columns else 0
        feat_n = len([c for c in df.columns if c not in NON_FEATURE_COLS and c != "run_id"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Total runs", len(df))
        m2.metric("Labeled", labeled_n)
        m3.metric("Feature columns", feat_n)

        display_cols = [c for c in [
            "run_id", "label", "overall_quality",
            "mesh_fineness", "num_inflation_layers", "cell_count",
            "nonOrthogonality_average", "skewness_average",
        ] if c in df.columns]
        st.dataframe(df[display_cols], width="stretch", hide_index=True)

        st.download_button(
            "⬇️  Download full CSV",
            df.to_csv(index=False).encode(),
            file_name="mesh_dataset.csv",
            mime="text/csv",
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORE DATA
# ════════════════════════════════════════════════════════════════════════════════

with tab_explore:
    df = st.session_state.df

    if df.empty:
        st.info("No data yet — go to **Parse Logs** to add runs.")
    else:
        labeled_n = int(df["label"].notna().sum()) if "label" in df.columns else 0
        feat_n = len([c for c in df.columns if c not in NON_FEATURE_COLS and c != "run_id"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Runs", len(df))
        m2.metric("Labeled", labeled_n)
        m3.metric("Features", feat_n)

        # ── Overall quality bar chart ─────────────────────────────────────────
        if "overall_quality" in df.columns:
            st.subheader("Overall mesh quality by run")

            chart_df = df[["run_id", "overall_quality"]].copy()
            chart_df["label"] = df["label"].fillna("unlabeled") if "label" in df.columns else "unlabeled"

            fig_bar = px.bar(
                chart_df.sort_values("overall_quality", ascending=False),
                x="run_id",
                y="overall_quality",
                color="label",
                color_discrete_map=LABEL_COLORS,
                category_orders={"label": LABEL_ORDER},
                labels={"run_id": "Run", "overall_quality": "Overall Quality Score"},
                height=350,
                text_auto=".3f",
            )
            fig_bar.add_hline(
                y=0.65, line_dash="dot", line_color="#94a3b8",
                annotation_text="0.65 reference", annotation_position="top right",
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(
                yaxis_range=[0, 1.15],
                yaxis_title="Score (0–1, higher = better)",
                xaxis_title=None,
                legend_title="Label",
                margin=dict(t=20, b=10),
            )
            st.plotly_chart(fig_bar, width="stretch")

        # ── Key metrics table ─────────────────────────────────────────────────
        st.divider()
        st.subheader("Key metrics comparison")

        avail = [m for m in HIGHLIGHT_METRICS if m in df.columns]
        if avail:
            tbl = df[["run_id"] + (["label"] if "label" in df.columns else []) + avail].copy()
            tbl.columns = [
                c.replace("nonOrthogonality_", "nonOrtho·")
                 .replace("skewness_", "skew·")
                 .replace("aspectRatio_", "AR·")
                 .replace("tetAspectRatio_", "tetAR·")
                 .replace("volumeRatio_", "volR·")
                for c in tbl.columns
            ]
            numeric_cols_in_tbl = [c for c in tbl.columns if c not in ("run_id", "label")]
            st.dataframe(
                tbl.set_index("run_id").style.background_gradient(
                    subset=numeric_cols_in_tbl, cmap="RdYlGn_r", axis=0
                ).format({c: "{:.4f}" for c in numeric_cols_in_tbl}),
                width="stretch",
            )

        # ── Radar chart ───────────────────────────────────────────────────────
        st.divider()
        st.subheader("Metric profile — radar chart")
        st.caption("Each axis is normalized to [0, 1] within the current dataset. Lower is better for most metrics.")

        radar_cols = [m for m in RADAR_METRICS if m in df.columns]
        if len(radar_cols) >= 4:
            radar_df = df[["run_id"] + radar_cols].copy()
            for col in radar_cols:
                lo, hi = radar_df[col].min(), radar_df[col].max()
                radar_df[col] = (radar_df[col] - lo) / (hi - lo) if hi != lo else 0.5

            theta = [
                c.replace("nonOrthogonality_", "nonOrtho·")
                 .replace("skewness_", "skew·")
                 .replace("aspectRatio_", "AR·")
                 .replace("tetAspectRatio_", "tetAR·")
                 .replace("volumeRatio_", "volR·")
                 .replace("tetEdgeRatio_", "tetEdge·")
                for c in radar_cols
            ]

            palette = px.colors.qualitative.Set2
            fig_radar = go.Figure()
            for i, (_, run_row) in enumerate(radar_df.iterrows()):
                vals = [run_row[c] for c in radar_cols]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=theta + [theta[0]],
                    fill="toself",
                    name=run_row["run_id"],
                    line_color=palette[i % len(palette)],
                    opacity=0.75,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                height=440,
                margin=dict(t=20, b=10),
            )
            st.plotly_chart(fig_radar, width="stretch")

        # ── Feature distribution ──────────────────────────────────────────────
        if len(df) >= 2:
            st.divider()
            st.subheader("Feature distribution")

            numeric_cols = sorted([
                c for c in df.columns
                if c not in NON_FEATURE_COLS and c != "run_id"
                and pd.api.types.is_numeric_dtype(df[c])
            ])
            default_idx = numeric_cols.index("overall_quality") if "overall_quality" in numeric_cols else 0
            sel = st.selectbox("Select a feature", numeric_cols, index=default_idx)

            color_by = "label" if "label" in df.columns and df["label"].notna().any() else None
            fig_hist = px.histogram(
                df if color_by is None else df.dropna(subset=["label"]),
                x=sel,
                color=color_by,
                color_discrete_map=LABEL_COLORS if color_by else None,
                barmode="overlay",
                nbins=25,
                opacity=0.75,
                height=280,
            )
            fig_hist.update_layout(margin=dict(t=10, b=10), xaxis_title=sel)
            st.plotly_chart(fig_hist, width="stretch")


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
            st.warning(
                f"Only **{len(labeled_df)}** labeled row(s). "
                f"Add **{6 - len(labeled_df)}** more to enable cross-validation. "
                "Aim for 15–30 runs for meaningful accuracy estimates."
            )
        else:
            feature_cols = [c for c in labeled_df.columns if c not in NON_FEATURE_COLS and c != "run_id"]
            feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(labeled_df[c])]
            feature_cols = [c for c in feature_cols if labeled_df[c].notna().any()]

            X = labeled_df[feature_cols].fillna(labeled_df[feature_cols].median())
            y = labeled_df["label"]

            # Summary strip
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Labeled runs", len(labeled_df))
            m2.metric("Features", len(feature_cols))
            m3.metric("Classes", y.nunique())
            m4.metric("CV folds", min(5, len(labeled_df)))

            # Class distribution
            st.subheader("Class distribution")
            dist_df = y.value_counts().reset_index()
            dist_df.columns = ["label", "count"]
            fig_dist = px.bar(
                dist_df, x="label", y="count",
                color="label", color_discrete_map=LABEL_COLORS,
                text_auto=True, height=220,
            )
            fig_dist.update_layout(showlegend=False, margin=dict(t=10, b=10), xaxis_title=None)
            fig_dist.update_traces(textposition="outside")
            st.plotly_chart(fig_dist, width="stretch")

            st.divider()

            if st.button("🚀  Train & evaluate all models", type="primary"):
                cv = min(5, len(labeled_df))
                models = {
                    "Gaussian Naive Bayes": Pipeline([
                        ("scale", StandardScaler()), ("nb", GaussianNB()),
                    ]),
                    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
                    "Logistic Regression": Pipeline([
                        ("scale", StandardScaler()),
                        ("lr", LogisticRegression(max_iter=2000)),
                    ]),
                }

                results = []
                with st.spinner("Training…"):
                    for name, model in models.items():
                        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
                        results.append({
                            "Model": name,
                            "CV Accuracy": float(scores.mean()),
                            "± Std": float(scores.std()),
                        })

                results_df = pd.DataFrame(results).sort_values("CV Accuracy", ascending=False)

                # Model comparison
                st.subheader("Model comparison")
                fig_models = go.Figure(go.Bar(
                    x=results_df["Model"],
                    y=results_df["CV Accuracy"],
                    error_y=dict(type="data", array=results_df["± Std"].tolist()),
                    marker_color=["#6366f1", "#22c55e", "#f59e0b"][: len(results_df)],
                    text=[f"{v:.3f}" for v in results_df["CV Accuracy"]],
                    textposition="outside",
                ))
                fig_models.update_layout(
                    yaxis_range=[0, 1.15],
                    yaxis_title="CV Accuracy",
                    xaxis_title=None,
                    height=300,
                    margin=dict(t=10, b=10),
                )
                st.plotly_chart(fig_models, width="stretch")

                # Confusion matrix — best model
                st.divider()
                best_name = results_df.iloc[0]["Model"]
                best_model = models[best_name]
                strat = y if y.value_counts().min() >= 2 else None
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=strat
                )
                best_model.fit(X_tr, y_tr)
                y_pred = best_model.predict(X_te)
                labels_present = sorted(y.unique())
                cm = confusion_matrix(y_te, y_pred, labels=labels_present)

                st.subheader(f"Confusion matrix — {best_name}")
                fig_cm = px.imshow(
                    cm,
                    x=labels_present,
                    y=labels_present,
                    color_continuous_scale="Blues",
                    text_auto=True,
                    labels=dict(x="Predicted", y="True", color="Count"),
                    height=340,
                )
                fig_cm.update_layout(margin=dict(t=20, b=10))
                st.plotly_chart(fig_cm, width="stretch")

                # Feature importance — Random Forest
                st.divider()
                st.subheader("Top 15 predictive features — Random Forest")
                rf = RandomForestClassifier(n_estimators=200, random_state=42)
                rf.fit(X, y)
                imp_df = (
                    pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_})
                    .sort_values("importance", ascending=False)
                    .head(15)
                )
                fig_imp = px.bar(
                    imp_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    color="importance",
                    color_continuous_scale="Blues",
                    height=460,
                    labels={"importance": "Importance", "feature": "Feature"},
                )
                fig_imp.update_layout(
                    yaxis=dict(autorange="reversed"),
                    showlegend=False,
                    coloraxis_showscale=False,
                    margin=dict(t=10, b=10),
                )
                st.plotly_chart(fig_imp, width="stretch")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — FORCE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

CONVERGENCE_THRESHOLDS = {
    "Converged":     0.01,   # CV < 1 %
    "Marginal":      0.05,   # CV < 5 %
    # else → Not converged
}

def convergence_status(cv: float) -> tuple[str, str]:
    """Return (label, colour) for a coefficient of variation value."""
    if cv < CONVERGENCE_THRESHOLDS["Converged"]:
        return "Converged ✅", "#22c55e"
    if cv < CONVERGENCE_THRESHOLDS["Marginal"]:
        return "Marginal ⚠️", "#f59e0b"
    return "Not converged ❌", "#ef4444"


with tab_forces:
    st.subheader("Force & moment coefficient analysis")
    st.caption(
        "Upload the force-coefficient CSV exported from SimScale. "
        "The app will plot every coefficient over time and assess convergence."
    )

    force_file = st.file_uploader(
        "Upload force-coefficient CSV",
        type=["csv"],
        key="force_csv",
        help="Expected columns: a time column (s / iteration) + one or more FORCE_COEFFICIENT_* columns.",
    )

    if force_file is not None:
        # ── Load & detect columns ─────────────────────────────────────────────
        try:
            fdf = pd.read_csv(force_file)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            fdf = None

        if fdf is not None:
            fdf.columns = fdf.columns.str.strip()

            # Auto-detect time column: prefer "Time", else first numeric col
            time_candidates = [c for c in fdf.columns if "time" in c.lower() or "iteration" in c.lower()]
            time_col = time_candidates[0] if time_candidates else fdf.columns[0]

            coeff_cols = [
                c for c in fdf.columns
                if c != time_col and pd.api.types.is_numeric_dtype(fdf[c])
            ]

            if not coeff_cols:
                st.error("No numeric coefficient columns found after the time column.")
            else:
                fdf = fdf[[time_col] + coeff_cols].dropna()

                # ── Column selector ───────────────────────────────────────────
                st.divider()
                selected_cols = st.multiselect(
                    "Coefficients to display",
                    coeff_cols,
                    default=coeff_cols,
                )
                if not selected_cols:
                    selected_cols = coeff_cols

                # Convergence window slider
                window_pct = st.slider(
                    "Convergence window (% of simulation from the end)",
                    min_value=5, max_value=50, value=20, step=5,
                    help="Stats are computed over this trailing portion of the time series.",
                )
                n_window = max(2, int(len(fdf) * window_pct / 100))
                window_df = fdf.tail(n_window)

                # ── Time-series plot ──────────────────────────────────────────
                st.divider()
                st.subheader("Coefficient history")

                plot_df = fdf[[time_col] + selected_cols].melt(
                    id_vars=time_col, var_name="Coefficient", value_name="Value"
                )
                fig_lines = px.line(
                    plot_df,
                    x=time_col,
                    y="Value",
                    color="Coefficient",
                    height=420,
                    labels={time_col: time_col, "Value": "Coefficient value"},
                )

                # Shade the convergence window
                x_start = float(fdf[time_col].iloc[-n_window])
                x_end   = float(fdf[time_col].iloc[-1])
                fig_lines.add_vrect(
                    x0=x_start, x1=x_end,
                    fillcolor="#6366f1", opacity=0.08,
                    line_width=0,
                    annotation_text=f"convergence window ({window_pct}%)",
                    annotation_position="top left",
                )
                # Rolling mean overlays (dashed, same colour, labelled)
                roll_n = max(3, len(fdf) // 10)
                palette = px.colors.qualitative.Plotly
                for i, col in enumerate(selected_cols):
                    rolled = fdf[col].rolling(roll_n, center=True).mean()
                    fig_lines.add_scatter(
                        x=fdf[time_col], y=rolled,
                        mode="lines",
                        line=dict(dash="dash", width=1.5, color=palette[i % len(palette)]),
                        name=f"{col} (avg)",
                        showlegend=True,
                        opacity=0.6,
                    )
                fig_lines.update_layout(margin=dict(t=20, b=10))
                st.plotly_chart(fig_lines, width="stretch")

                # ── Convergence table ─────────────────────────────────────────
                st.divider()
                st.subheader("Convergence assessment")
                st.caption(
                    f"Statistics computed over the last **{window_pct}%** "
                    f"({n_window} time steps) of the simulation."
                )

                rows = []
                for col in selected_cols:
                    window_vals = window_df[col]
                    mean_val  = window_vals.mean()
                    std_val   = window_vals.std()
                    cv        = abs(std_val / mean_val) if mean_val != 0 else float("inf")
                    total_chg = abs(fdf[col].iloc[-1] - fdf[col].iloc[0])
                    status, _ = convergence_status(cv)
                    rows.append({
                        "Coefficient":   col,
                        "Final mean":    mean_val,
                        "Std dev":       std_val,
                        "CV (std/mean)": cv,
                        "Total change":  total_chg,
                        "Status":        status,
                    })

                conv_df = pd.DataFrame(rows)

                # Colour the Status column
                def colour_status(val: str) -> str:
                    if "Converged ✅" == val:
                        return "background-color: #dcfce7"
                    if "Marginal" in val:
                        return "background-color: #fef9c3"
                    return "background-color: #fee2e2"

                fmt = {
                    "Final mean":    "{:.4e}",
                    "Std dev":       "{:.4e}",
                    "CV (std/mean)": "{:.4f}",
                    "Total change":  "{:.4e}",
                }
                st.dataframe(
                    conv_df.style
                        .format(fmt)
                        .applymap(colour_status, subset=["Status"]),
                    width="stretch",
                    hide_index=True,
                )

                n_conv = sum(1 for r in rows if "Converged ✅" == r["Status"])
                n_marg = sum(1 for r in rows if "Marginal" in r["Status"])
                n_bad  = sum(1 for r in rows if "Not converged" in r["Status"])
                final_means = {r["Coefficient"]: r["Final mean"] for r in rows}

                # ── Derived aerodynamic metrics ───────────────────────────────
                st.divider()
                st.subheader("Aerodynamic metrics")

                def coeff_suffix(col: str) -> str:
                    return col.split("_")[-1].upper()

                cd_col  = next((c for c in selected_cols if coeff_suffix(c) == "CD"),  None)
                cl_col  = next((c for c in selected_cols if coeff_suffix(c) == "CL"),  None)
                clf_col = next((c for c in selected_cols if coeff_suffix(c) == "CLF"), None)
                clr_col = next((c for c in selected_cols if coeff_suffix(c) == "CLR"), None)
                cm_col  = next((c for c in selected_cols if coeff_suffix(c) == "CM"),  None)

                metric_cards: list[tuple[str, str, str]] = []  # (label, value, help)

                if cd_col:
                    metric_cards.append(("Cd (final mean)", f"{final_means[cd_col]:.4e}", "Drag coefficient — resistance to motion. Lower = less drag."))
                if cl_col:
                    cl_v = final_means[cl_col]
                    sign_note = "downforce ✅" if cl_v < 0 else "lift"
                    metric_cards.append(("Cl (final mean)", f"{cl_v:.4e}", f"Total lift coefficient ({sign_note}). Negative = downforce, which increases grip."))
                if cd_col and cl_col and final_means[cd_col] != 0:
                    ld = abs(final_means[cl_col] / final_means[cd_col])
                    metric_cards.append(("L/D ratio", f"{ld:.2f}", "Lift-to-drag efficiency. Higher = more aerodynamic force per unit drag."))
                if clf_col and clr_col:
                    total = final_means[clf_col] + final_means[clr_col]
                    if total != 0:
                        bal = final_means[clf_col] / total * 100
                        metric_cards.append(("Aero balance", f"{bal:.1f}% front", "Front/(front+rear) downforce split. ~45–55% front is typical for balanced handling."))
                if cm_col:
                    metric_cards.append(("Cm (final mean)", f"{final_means[cm_col]:.4e}", "Pitching moment — nose-up/down tendency. Near 0 = balanced pitch forces."))

                if metric_cards:
                    cols_m = st.columns(len(metric_cards))
                    for i, (lbl, val, hlp) in enumerate(metric_cards):
                        cols_m[i].metric(lbl, val, help=hlp)

                # ── Trend & oscillation analysis ──────────────────────────────
                st.divider()
                st.subheader("Trend analysis")
                st.caption(
                    f"Linear drift and oscillation computed over the convergence "
                    f"window (last {window_pct}%, {n_window} steps). "
                    "Dashed lines on the plot above show the 10-point rolling mean."
                )

                trend_rows = []
                for col in selected_cols:
                    x_w = window_df[time_col].values.astype(float)
                    y_w = window_df[col].values.astype(float)
                    slope = float(np.polyfit(x_w - x_w.mean(), y_w, 1)[0])
                    mean_abs = abs(final_means[col])
                    drift_pct = abs(slope) / mean_abs * 100 if mean_abs > 0 else float("inf")

                    diffs = np.diff(y_w)
                    sign_chg = int(np.sum(np.diff(np.sign(diffs)) != 0))
                    oscillating = sign_chg > len(diffs) * 0.4

                    direction = "Rising ↑" if slope > 1e-12 else ("Falling ↓" if slope < -1e-12 else "Flat →")
                    trend_rows.append({
                        "Coefficient":              col,
                        "Slope (per time unit)":    slope,
                        "Drift % per 100 units":    drift_pct,
                        "Trend":                    direction,
                        "Oscillating?":             "Yes ↕" if oscillating else "No →",
                    })

                trend_df = pd.DataFrame(trend_rows)
                st.dataframe(
                    trend_df.style.format({
                        "Slope (per time unit)":  "{:.4e}",
                        "Drift % per 100 units":  "{:.2f}",
                    }),
                    width="stretch",
                    hide_index=True,
                )

                # ── CFD insights ──────────────────────────────────────────────
                st.divider()
                st.subheader("What the data is telling you")

                insights: list[tuple[str, str]] = []

                # Convergence status
                if n_bad == 0 and n_marg == 0:
                    insights.append(("success", f"All {len(selected_cols)} coefficient(s) have **converged** (CV < 1% in the last {window_pct}% of the run). It is safe to use the final mean values for design decisions."))
                elif n_bad == 0:
                    insights.append(("warning", f"{n_conv} converged, {n_marg} are **marginal**. Run the simulation for roughly {int(len(fdf) * 0.5)} more time steps and re-check before extracting final values."))
                else:
                    insights.append(("error", f"{n_bad} coefficient(s) are **still drifting** (CV ≥ 5%). The simulation has not reached steady state — do not use these values yet. Extend the run time and monitor whether the drift rate is decreasing."))

                # Still-drifting coefficients
                fast_drift = [r for r in trend_rows if r["Drift % per 100 units"] > 5]
                if fast_drift:
                    names = ", ".join(r["Coefficient"] for r in fast_drift)
                    insights.append(("error", f"**Fast drift detected** in {names}: values are changing >5% per 100 time units. The solver has not settled — keep running."))

                # L/D insight
                ld_card = next((c for c in metric_cards if c[0] == "L/D ratio"), None)
                if ld_card:
                    ld_val = float(ld_card[1])
                    if ld_val > 10:
                        insights.append(("success", f"**L/D = {ld_val:.1f}** — excellent aerodynamic efficiency. The geometry generates a lot of downforce/lift for its drag penalty."))
                    elif ld_val > 3:
                        insights.append(("info", f"**L/D = {ld_val:.1f}** — moderate efficiency. There may be drag-reduction opportunities (smoothing wake geometry, reducing frontal area) without sacrificing much downforce."))
                    else:
                        insights.append(("warning", f"**L/D = {ld_val:.1f}** — drag is high relative to lift. Review bluff bodies, separation zones, or high-angle surfaces in the geometry."))

                # Aero balance insight
                bal_card = next((c for c in metric_cards if c[0] == "Aero balance"), None)
                if bal_card:
                    bal_val = float(bal_card[1].split("%")[0])
                    if 42 <= bal_val <= 58:
                        insights.append(("success", f"**Aero balance {bal_val:.1f}% front** — well centred. The downforce is distributed evenly between axles, supporting neutral handling balance."))
                    elif bal_val > 58:
                        insights.append(("warning", f"**Aero balance {bal_val:.1f}% front** — front-heavy. Excess front downforce can cause understeer at high speed. Consider reducing front diffuser angle or adding rear wing."))
                    else:
                        insights.append(("warning", f"**Aero balance {bal_val:.1f}% front** — rear-heavy. Excess rear downforce can cause oversteer. Consider reducing rear wing angle or increasing front splitter size."))

                # Pitching moment insight
                if cm_col:
                    cm_v = final_means[cm_col]
                    cd_ref = abs(final_means[cd_col]) if cd_col else 1
                    if abs(cm_v) < cd_ref * 0.5:
                        insights.append(("info", f"**Pitching moment Cm ≈ {cm_v:.3e}** — relatively small. The geometry produces balanced pitch forces with no strong nose-up or nose-down tendency."))
                    elif cm_v > 0:
                        insights.append(("warning", f"**Pitching moment Cm = {cm_v:.3e} (positive / nose-up)**. At high speed this transfers load to the rear axle and reduces front grip. Check front-end geometry."))
                    else:
                        insights.append(("warning", f"**Pitching moment Cm = {cm_v:.3e} (negative / nose-down)**. Increased front load — monitor for front tyre overloading or understeer at high speed."))

                # Oscillation insight
                osc = [r["Coefficient"] for r in trend_rows if "Yes" in r["Oscillating?"]]
                if osc:
                    insights.append(("warning", f"**Oscillating signal** in: {', '.join(osc)}. This can indicate: unsteady separated flow, vortex shedding, or a time step that is too coarse. Check mesh refinement in wake regions and consider a finer time step."))

                # Cl sign check
                if cl_col:
                    if final_means[cl_col] > 0:
                        insights.append(("warning", "**Cl is positive (lift, not downforce)**. For ground-vehicle aerodynamics this increases load transfer and reduces grip. Verify this is the intended configuration — a negative Cl (downforce) is usually desirable."))

                for severity, msg in insights:
                    getattr(st, severity)(msg)

                # ── Final-value summary ───────────────────────────────────────
                st.divider()
                with st.expander("Final values — mean ± std over convergence window", expanded=True):
                    for r in rows:
                        status_icon = "✅" if "Converged" in r["Status"] else ("⚠️" if "Marginal" in r["Status"] else "❌")
                        st.markdown(
                            f"{status_icon} **{r['Coefficient']}**: "
                            f"`{r['Final mean']:.6e}` ± `{r['Std dev']:.2e}`"
                        )

                # ── Reference glossary ────────────────────────────────────────
                with st.expander("📖 Coefficient reference guide"):
                    st.markdown("""
| Coefficient | What it measures | Good direction | Typical range |
|---|---|---|---|
| **Cd** | Drag — resistance to forward motion | Lower = less drag | 0.1 – 0.5 (streamlined bodies) |
| **Cl** | Total lift/downforce | Negative = downforce (more grip) | −3 to +1 depending on config |
| **Clf** | Front-axle lift component | Negative = front downforce | — |
| **Clr** | Rear-axle lift component | Negative = rear downforce | — |
| **Cm** | Pitching moment (nose-up/down tendency) | Close to 0 = balanced | −0.5 to +0.5 |
| **L/D** | Aerodynamic efficiency (Cl / Cd) | Higher = more force per drag | 1 – 20+ |
| **Aero balance** | Front ÷ (Front + Rear) downforce | 42 – 58% front = neutral | — |
""")
                    st.markdown("""
**Convergence criteria:**
- ✅ **Converged** — coefficient of variation (std ÷ |mean|) < 1% in the window
- ⚠️ **Marginal** — CV between 1% and 5%
- ❌ **Not converged** — CV > 5%

**Oscillation flag:** triggered when >40% of consecutive differences alternate sign inside the window, suggesting unsteady or periodically separated flow.
""")
    else:
        st.info("Upload a force-coefficient CSV above to get started.")
