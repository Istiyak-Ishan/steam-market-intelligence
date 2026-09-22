"""
model_lab.py -- Model Inspector: diagnostics, feature importance, and evaluation metrics.

Displays pre-computed training metrics, permutation importance, confusion matrix,
and classification report for the precision ML pipeline.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from src.config import MODELS_DIR, ACCENT_COLORS, PLOTLY_BG_COLOR, PLOTLY_PAPER_BG


def _layout(**kw) -> dict:
    base = dict(
        template="plotly_dark",
        plot_bgcolor=PLOTLY_BG_COLOR,
        paper_bgcolor=PLOTLY_PAPER_BG,
        font=dict(color="#e0e0e0", family="Inter, sans-serif", size=12),
        margin=dict(t=50, r=16, b=50, l=150),
    )
    base.update(kw)
    return base


def render(df=None, models=None, **kwargs) -> None:
    hide_header = kwargs.get("hide_header", False)
    if not hide_header:
        st.markdown(
            '<div class="hero-header"><div class="hero-title">🧪 Model Inspector</div>'
            '<div class="hero-subtitle">Evaluation metrics, feature importance, and classification diagnostics for all trained models.</div></div>',
            unsafe_allow_html=True,
        )

    report_path = Path(MODELS_DIR) / "training_report.md"
    perm_path = Path(MODELS_DIR) / "permutation_importance.csv"
    cm_path = Path(MODELS_DIR) / "confusion_matrix.csv"
    cr_path = Path(MODELS_DIR) / "classification_report.txt"

    # ── Tab layout ────────────────────────────────────────────────────────
    tab_metrics, tab_importance, tab_clf = st.tabs([
        "Model Metrics", "Feature Importance", "Classifier Diagnostics"
    ])

    # ── TAB 1: Model Metrics ──────────────────────────────────────────────
    with tab_metrics:
        st.markdown('<div class="section-header">Training Report (Test Set Evaluation)</div>', unsafe_allow_html=True)
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                report_text = f.read()
            st.markdown(report_text)
        else:
            st.warning("Training report not found. Run `python scripts/train_pipeline.py` first.")

        st.markdown("---")
        st.markdown('<div class="section-header">Model Architecture</div>', unsafe_allow_html=True)
        arch_data = {
            "Model": [
                "Price Sweetspot", "Review Score", "Value Score",
                "Ownership", "Price Tier Classifier"
            ],
            "Algorithm": [
                "HistGradientBoostingRegressor", "HistGradientBoostingRegressor",
                "HistGradientBoostingRegressor", "HistGradientBoostingRegressor",
                "HistGradientBoostingClassifier"
            ],
            "Target": [
                "price (USD)", "quality_score (0-100)",
                "quality_score / price", "log1p(owners)",
                "Price Tier (Budget/Mid/Premium/AAA)"
            ],
            "Features": ["11 base", "11 base + price", "11 base", "11 base + price", "11 base"],
            "Split": ["80/20", "80/20", "80/20", "80/20", "80/20 stratified"],
        }
        st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

    # ── TAB 2: Feature Importance ─────────────────────────────────────────
    with tab_importance:
        st.markdown('<div class="section-header">Permutation Importance (Ownership Model)</div>', unsafe_allow_html=True)
        st.caption("Measures each feature's impact by shuffling its values and measuring the drop in R\u00b2 on the test set. Higher = more important.")

        if perm_path.exists():
            perm_df = pd.read_csv(perm_path).sort_values("importance_mean", ascending=True)
            fig = px.bar(
                perm_df, x="importance_mean", y="feature",
                orientation="h",
                error_x="importance_std",
                color="importance_mean",
                color_continuous_scale="Viridis",
                labels={"importance_mean": "Mean Importance (R\u00b2 Drop)", "feature": "Feature"},
            )
            fig.update_layout(**_layout(
                title="Permutation Feature Importance (Top Ownership Model, Test Set)",
                coloraxis_showscale=False,
            ))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown('<div class="section-header">Correlation-Based Feature Analysis</div>', unsafe_allow_html=True)
            if df is not None:
                numeric_cols = [
                    "price", "review_score_pct", "peak_ccu", "average_playtime_forever",
                    "languages_count", "categories_count", "genre_count", "age_by_years",
                    "owners_mid", "total_review", "recommendations",
                ]
                available = [c for c in numeric_cols if c in df.columns]
                corr = df[available].corr()

                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=[c.replace("_", " ").title() for c in corr.columns],
                    y=[c.replace("_", " ").title() for c in corr.index],
                    colorscale="RdBu_r",
                    zmin=-1, zmax=1,
                    text=corr.round(2).values,
                    texttemplate="%{text}",
                    textfont=dict(size=9),
                ))
                fig_corr.update_layout(**_layout(
                    title="Feature Correlation Matrix",
                    height=550,
                    margin=dict(t=50, r=16, b=50, l=150),
                ))
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Permutation importance file not found. Run `python scripts/train_pipeline.py`.")

    # ── TAB 3: Classifier Diagnostics ─────────────────────────────────────
    with tab_clf:
        st.markdown('<div class="section-header">Price Tier Classifier: Confusion Matrix</div>', unsafe_allow_html=True)

        if cm_path.exists():
            cm_df = pd.read_csv(cm_path, index_col=0)
            tier_names = cm_df.columns.tolist()

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm_df.values,
                x=tier_names,
                y=tier_names,
                colorscale="Blues",
                text=cm_df.values,
                texttemplate="%{text}",
                textfont=dict(size=14),
            ))
            fig_cm.update_layout(**_layout(
                title="Confusion Matrix (Test Set)",
                xaxis_title="Predicted Tier",
                yaxis_title="Actual Tier",
                height=450,
                margin=dict(t=50, r=16, b=60, l=100),
            ))
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.warning("Confusion matrix not found.")

        st.markdown("---")
        st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
        if cr_path.exists():
            with open(cr_path, "r") as f:
                cr_text = f.read()
            st.code(cr_text, language="text")
        else:
            st.warning("Classification report not found.")