"""
model_lab.py -- ML Model Laboratory page.

Features:
  1. Price-Value Predictor
  2. Price Tier Classifier
  3. Fair Price Advisor
  4. Model Metrics (Actual vs Predicted, Confusion Matrix)
  5. Feature Importance (Native tree SHAP/Gini importance)
"""
from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from sklearn.metrics import mean_absolute_error, confusion_matrix
import math

import src.visualization as vz

from src.model_loader import (
    predict_value_score, predict_price_tier, 
    load_price_value_model, load_price_tier_model, validate_model_features_match
)
from src.config import MODEL_FEATURES, CLASSIFIER_TIERS, ACCENT_COLORS, PLOTLY_BG_COLOR, PLOTLY_PAPER_BG
from src.feature_engineering import build_model_input

@st.cache_data
def _compute_metrics(df: pd.DataFrame) -> dict:
    """Compute and cache model metrics on the current dataset."""
    valid_df = df.dropna(subset=MODEL_FEATURES).copy()
    if valid_df.empty:
        return {}
        
    reg_model = load_price_value_model()
    clf_model = load_price_tier_model()
    
    metrics = {}
    
    # Regression Metrics
    if "value_score_calc" in valid_df.columns:
        valid_reg = valid_df.dropna(subset=["value_score_calc"])
        X_reg = valid_reg[MODEL_FEATURES]
        y_true_reg = valid_reg["value_score_calc"].values
        y_pred_reg = reg_model.predict(X_reg)
        
        mae = mean_absolute_error(y_true_reg, y_pred_reg)
        rmse = math.sqrt(np.mean((y_true_reg - y_pred_reg)**2))
        
        metrics["reg_mae"] = mae
        metrics["reg_rmse"] = rmse
        
        # Sample for plotting
        sample_size = min(5000, len(y_true_reg))
        idx = np.random.choice(len(y_true_reg), sample_size, replace=False)
        metrics["reg_plot_data"] = pd.DataFrame({
            "Actual": y_true_reg[idx],
            "Predicted": y_pred_reg[idx],
            "Residual": (y_true_reg - y_pred_reg)[idx]
        })
        
    # Classification Metrics
    if "price_tier" in valid_df.columns:
        valid_clf = valid_df[valid_df["price_tier"] != "Free"].dropna(subset=["price_tier"])
        X_clf = valid_clf[MODEL_FEATURES]
        y_true_clf = valid_clf["price_tier"].values
        y_pred_clf = clf_model.predict(X_clf)
        
        classes = clf_model.classes_
        cm = confusion_matrix(y_true_clf, y_pred_clf, labels=classes)
        metrics["clf_cm"] = cm
        metrics["clf_classes"] = classes
        metrics["clf_acc"] = np.mean(y_true_clf == y_pred_clf)

    return metrics


def _model_inputs(key_prefix: str) -> dict:
    col1, col2, col3 = st.columns(3)
    with col1:
        review_pct  = st.slider("Review Score (%)", 0, 100, 75, key=f"{key_prefix}_review")
        age_years   = st.number_input("Game Age (years)", 0.0, 30.0, 2.0, step=0.5, key=f"{key_prefix}_age")
        languages   = st.number_input("Languages Supported", 1, 100, 5, key=f"{key_prefix}_lang")
        categories  = st.number_input("Steam Categories Count", 1, 30, 4, key=f"{key_prefix}_cat")
    with col2:
        peak_ccu    = st.number_input("Peak CCU", 0, 5000000, 1000, step=500, key=f"{key_prefix}_ccu")
        total_rev   = st.number_input("Total Reviews", 0, 5000000, 200, key=f"{key_prefix}_rev")
        playtime    = st.number_input("Avg Playtime (minutes)", 0, 100000, 300, key=f"{key_prefix}_pt")
        genre_count = st.number_input("Genre Count", 1, 10, 2, key=f"{key_prefix}_gc")
    with col3:
        audio_lang  = st.number_input("Full Audio Languages", 0, 20, 1, key=f"{key_prefix}_audio")
        is_indie    = st.checkbox("Indie Game", value=True, key=f"{key_prefix}_indie")
        is_casual   = st.checkbox("Casual Genre", value=False, key=f"{key_prefix}_casual")
        cat_sp      = st.checkbox("Single-Player", value=True, key=f"{key_prefix}_sp")

    return {
        "quality_score":              float(review_pct),
        "age_by_years":               float(age_years),
        "categories_count":           float(categories),
        "languages_count":            float(languages),
        "peak_ccu":                   float(peak_ccu),
        "log_reviews":                float(np.log1p(total_rev)),
        "genre_casual":               1.0 if is_casual else 0.0,
        "genre_count":                float(genre_count),
        "full_audio_languages_count": float(audio_lang),
        "is_indie":                   1.0 if is_indie else 0.0,
        "average_playtime_forever":   float(playtime),
        "cat_single_player":          1.0 if cat_sp else 0.0,
    }


def _proba_bar(proba: dict) -> go.Figure:
    df = pd.DataFrame(list(proba.items()), columns=["Tier", "Probability"])
    df["Pct"] = df["Probability"] * 100
    fig = px.bar(df, x="Pct", y="Tier", orientation="h",
                 text=df["Pct"].apply(lambda x: f"{x:.1f}%"),
                 color="Tier", color_discrete_sequence=ACCENT_COLORS,
                 labels={"Pct": "Probability (%)", "Tier": "Price Tier"})
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_range=[0, 110])
    return vz._apply_theme(fig, "")


def _fair_price_analysis(tier_pred: str, proba: dict, actual_price: float, df: pd.DataFrame) -> None:
    TIER_RANGES = {
        "Budget":    (0.99, 9.99),
        "Mid-range": (9.99, 29.99),
        "Premium":   (29.99, 59.99),
        "AAA":       (59.99, 79.99),
    }
    rng = TIER_RANGES.get(tier_pred, (0, 0))
    tier_df = df[df["price_tier"] == tier_pred]["price"].dropna()
    mkt_median = tier_df.median() if len(tier_df) > 0 else (rng[0] + rng[1]) / 2
    mkt_p25    = tier_df.quantile(0.25) if len(tier_df) > 0 else rng[0]
    mkt_p75    = tier_df.quantile(0.75) if len(tier_df) > 0 else rng[1]

    if actual_price < rng[0]:
        verdict, vcolor = "Potentially Underpriced for Predicted Tier", "#10B981"
        advice = f"Model predicts **{tier_pred}** tier (typical: ${rng[0]:.2f}-${rng[1]:.2f}). Your ${actual_price:.2f} is below the typical floor."
    elif actual_price > rng[1]:
        verdict, vcolor = "Potentially Overpriced for Predicted Tier", "#EF4444"
        advice = f"Model predicts **{tier_pred}** tier (${rng[0]:.2f}-${rng[1]:.2f}). Your ${actual_price:.2f} exceeds this range."
    else:
        verdict, vcolor = "Price Aligns With Predicted Tier", "#7C3AED"
        advice = f"Your ${actual_price:.2f} falls within the expected range for **{tier_pred}** games."

    st.markdown(f"""
    <div class="metric-card" style="border-color:{vcolor};">
        <div class="metric-value" style="color:{vcolor}; font-size:1.3rem;">{verdict}</div>
        <div style="margin-top:10px; color:#e0e0e0; font-size:0.95rem;">{advice}</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Your Price",      f"${actual_price:.2f}")
    with c2: st.metric("Predicted Tier",  tier_pred)
    with c3: st.metric("Tier Median",     f"${mkt_median:.2f}")
    with c4: st.metric("IQR Range",       f"${mkt_p25:.2f}-${mkt_p75:.2f}")

    st.markdown("""
    <div class="info-box">
        <strong>Interpretation note:</strong> The model predicts which tier a game's
        quality/engagement profile most resembles. This is an analytical signal,
        not a pricing recommendation or guarantee of commercial success.
    </div>
    """, unsafe_allow_html=True)


def render(df: pd.DataFrame, models: dict, hide_header: bool = False) -> None:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🔬 Model Lab</div>
        <div class="hero-subtitle">
            Interactive ML model interface -- predict value score, price tier,
            evaluate pricing alignment, and examine model metrics / feature importance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>Models:</strong> Random Forest Regressor (value score) and Random Forest Classifier
        (price tier) trained on commercial Steam titles.
        <strong>No causality is implied</strong> -- outputs are statistical associations.
    </div>
    """, unsafe_allow_html=True)

    tab_value, tab_tier, tab_fair, tab_metrics, tab_feat = st.tabs([
        "Price-Value Predictor", "Price Tier Classifier", "Fair Price Advisor", "Model Metrics", "Feature Importance"
    ])

    with tab_value:
        st.markdown("**Value Score** = quality score / price. The model predicts this ratio from game features.")
        st.markdown('<div class="section-header">Game Feature Inputs</div>', unsafe_allow_html=True)
        profile_v = _model_inputs("value")
        if st.button("Predict Value Score", key="btn_value"):
            try:
                val = predict_value_score(profile_v)
                pct_str = ""
                if "value_score_calc" in df.columns:
                    from scipy.stats import percentileofscore
                    pct = percentileofscore(df["value_score_calc"].dropna().values, val, kind="rank")
                    pct_str = f'<div style="color:#8b949e; margin-top:8px;">Better value than {pct:.1f}% of commercial titles in dataset</div>'

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{val:.3f} pts/$</div>
                    <div class="metric-label">Predicted Value Score (review quality points per $1)</div>
                    {pct_str}
                </div>
                """, unsafe_allow_html=True)
                
                reg_model = load_price_value_model()
                top_idx = np.argsort(reg_model.feature_importances_)[-3:][::-1]
                top_feats = [MODEL_FEATURES[i] for i in top_idx]
                st.info(f"**Primary Drivers (Global Importance):** {', '.join([f.replace('_', ' ').title() for f in top_feats])}")

                if "value_score_calc" in df.columns:
                    vs_plot = df["value_score_calc"].dropna().clip(upper=100)
                    fig = px.histogram(vs_plot, nbins=80,
                                       labels={"value": "Value Score", "count": "Games"},
                                       color_discrete_sequence=[ACCENT_COLORS[0]])
                    fig.add_vline(x=val, line_color=ACCENT_COLORS[1], line_width=2,
                                  annotation_text=f"Your game: {val:.2f}", annotation_position="top right")
                    fig = vz._apply_theme(fig, "Value Score Distribution (your prediction marked)")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with tab_tier:
        st.markdown("**Price Tier Classifier** predicts which tier a game's features most closely resemble.")
        st.markdown('<div class="section-header">Game Feature Inputs</div>', unsafe_allow_html=True)
        profile_t = _model_inputs("tier")
        if st.button("Predict Price Tier", key="btn_tier"):
            try:
                tier, proba = predict_price_tier(profile_t)
                COLORS = {"Budget": "#10B981", "Mid-range": "#3B82F6", "Premium": "#7C3AED", "AAA": "#F59E0B"}
                tc = COLORS.get(tier, "#7C3AED")

                st.markdown(f"""
                <div class="metric-card" style="border-color:{tc};">
                    <div class="metric-value" style="color:{tc};">{tier}</div>
                    <div class="metric-label">Predicted Natural Price Tier</div>
                </div>
                """, unsafe_allow_html=True)
                
                clf_model = load_price_tier_model()
                top_idx = np.argsort(clf_model.feature_importances_)[-3:][::-1]
                top_feats = [MODEL_FEATURES[i] for i in top_idx]
                st.info(f"**Primary Drivers (Global Importance):** {', '.join([f.replace('_', ' ').title() for f in top_feats])}")

                st.markdown('<div class="section-header">Tier Probabilities</div>', unsafe_allow_html=True)
                st.plotly_chart(_proba_bar(proba), use_container_width=True)

                if "price_tier" in df.columns:
                    tier_prices = df[df["price_tier"] == tier]["price"].dropna()
                    fig_tp = px.histogram(tier_prices.clip(upper=80), nbins=50,
                                          color_discrete_sequence=[tc],
                                          labels={"value": "Price (USD)", "count": "Games"})
                    fig_tp = vz._apply_theme(fig_tp, f"{tier} Tier Price Distribution ({len(tier_prices):,} games)")
                    st.plotly_chart(fig_tp, use_container_width=True)

                st.markdown("""
                <div class="info-box">
                    Tiers: Budget ($0.99-$10), Mid-range ($10-$30), Premium ($30-$60), AAA ($60+).
                    Free games excluded from training. Trained on commercial titles.
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Classifier failed: {e}")

    with tab_fair:
        st.markdown("Enter your feature profile and intended price to evaluate pricing alignment.")
        st.markdown('<div class="section-header">Game Feature Inputs</div>', unsafe_allow_html=True)
        profile_f = _model_inputs("fair")
        actual_price = st.number_input("Your Intended Price ($)", 0.0, 200.0, 14.99, step=0.50, key="fair_price")

        if st.button("Analyse Price Fit", key="btn_fair"):
            try:
                tier, proba = predict_price_tier(profile_f)
                _fair_price_analysis(tier, proba, actual_price, df)
                st.markdown('<div class="section-header">Full Probability Breakdown</div>', unsafe_allow_html=True)
                st.plotly_chart(_proba_bar(proba), use_container_width=True)
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    with tab_metrics:
        st.markdown('<div class="section-header">Model Metrics against Loaded Dataset</div>', unsafe_allow_html=True)
        st.markdown("Evaluating the loaded models against the current dataset (live evaluation).")
        
        with st.spinner("Computing metrics..."):
            metrics = _compute_metrics(df)
            
        if not metrics:
            st.warning("Insufficient valid data to compute model metrics.")
        else:
            col_rm, col_cm = st.columns(2)
            
            with col_rm:
                st.markdown("**Price-Value Regressor**")
                if "reg_mae" in metrics:
                    st.write(f"- Mean Absolute Error (MAE): **{metrics['reg_mae']:.3f} pts/$**")
                    st.write(f"- Root Mean Squared Error (RMSE): **{metrics['reg_rmse']:.3f} pts/$**")
                    
                    fig_scatter = px.scatter(
                        metrics["reg_plot_data"], x="Actual", y="Predicted",
                        opacity=0.3, color_discrete_sequence=[ACCENT_COLORS[0]]
                    )
                    fig_scatter.add_shape(
                        type="line", line=dict(dash="dash", color="white"),
                        x0=0, y0=0, x1=100, y1=100
                    )
                    fig_scatter = vz._apply_theme(fig_scatter, "Actual vs Predicted Value Score (Sampled)")
                    fig_scatter.update_layout(xaxis_range=[0, 100], yaxis_range=[0, 100])
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    fig_resid = px.histogram(
                        metrics["reg_plot_data"], x="Residual",
                        nbins=60, color_discrete_sequence=[ACCENT_COLORS[1]]
                    )
                    fig_resid = vz._apply_theme(fig_resid, "Residual Distribution (Actual - Predicted)")
                    fig_resid.update_layout(xaxis_range=[-50, 50])
                    st.plotly_chart(fig_resid, use_container_width=True)

            with col_cm:
                st.markdown("**Price Tier Classifier**")
                if "clf_cm" in metrics:
                    st.write(f"- Overall Accuracy: **{metrics['clf_acc']*100:.1f}%**")
                    
                    fig_cm = px.imshow(
                        metrics["clf_cm"],
                        x=metrics["clf_classes"],
                        y=metrics["clf_classes"],
                        labels=dict(x="Predicted Tier", y="Actual Tier", color="Count"),
                        text_auto=True, color_continuous_scale="Purples"
                    )
                    fig_cm = vz._apply_theme(fig_cm, "Confusion Matrix")
                    fig_cm.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_cm, use_container_width=True)

    with tab_feat:
        st.markdown('<div class="section-header">Feature Importances (Gini/Tree Native)</div>', unsafe_allow_html=True)
        st.markdown("Relative weight of each feature in the Random Forest models' decision making.")
        
        try:
            reg_model = load_price_value_model()
            clf_model = load_price_tier_model()
            
            # Regression importances
            reg_imp = reg_model.feature_importances_
            reg_df = pd.DataFrame({
                "Feature": [f.replace("_", " ").title() for f in MODEL_FEATURES],
                "Importance": reg_imp
            }).sort_values("Importance", ascending=True)
            
            # Classification importances
            clf_imp = clf_model.feature_importances_
            clf_df = pd.DataFrame({
                "Feature": [f.replace("_", " ").title() for f in MODEL_FEATURES],
                "Importance": clf_imp
            }).sort_values("Importance", ascending=True)
            
            c_f1, c_f2 = st.columns(2)
            with c_f1:
                fig_fi_reg = px.bar(
                    reg_df, x="Importance", y="Feature", orientation="h",
                    color_discrete_sequence=[ACCENT_COLORS[0]]
                )
                fig_fi_reg = vz._apply_theme(fig_fi_reg, "Value Score Regressor Importances")
                st.plotly_chart(fig_fi_reg, use_container_width=True)
                
            with c_f2:
                fig_fi_clf = px.bar(
                    clf_df, x="Importance", y="Feature", orientation="h",
                    color_discrete_sequence=[ACCENT_COLORS[1]]
                )
                fig_fi_clf = vz._apply_theme(fig_fi_clf, "Price Tier Classifier Importances")
                st.plotly_chart(fig_fi_clf, use_container_width=True)
                
        except Exception as e:
            st.error(f"Failed to load feature importances: {e}")