"""
CFD Mesh Quality Classifier — Streamlit Dashboard
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
    page_title="CFD Mesh Quality Classifier",
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

st.title("🌊 CFD Mesh Quality Classifier")
st.caption("Parse SimScale mesh logs · Explore quality metrics · Train & compare classifiers")

tab_parse, tab_explore, tab_train = st.tabs(
    ["📂  Parse Logs", "📊  Explore Data", "🤖  Train Models"]
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
