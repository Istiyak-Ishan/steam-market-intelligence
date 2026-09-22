"""
predict_tool.py — Predict Tool (Tab 5)

Single input form that runs all 3 ML predictions simultaneously:
  1. Predicted value_score with color-coded gauge
  2. Predicted price tier with confidence
  3. Fair vs. Overpriced verdict
"""
from __future__ import annotations

import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.model_loader import (
    predict_value_score, predict_price_tier, predict_fair_price,
    predict_price_sweetspot, predict_review_score, predict_ownership,
)


PLOTLY_BG = "rgba(0,0,0,0)"
PLOTLY_PAPER = "rgba(0,0,0,0)"


def _gauge(value: float, title: str, min_val: float, max_val: float,
           green_thresh: float, yellow_thresh: float) -> go.Figure:
    """Create a Plotly gauge chart."""
    if value >= green_thresh:
        bar_color = "#10B981"
    elif value >= yellow_thresh:
        bar_color = "#F59E0B"
    else:
        bar_color = "#EF4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 13, "color": "#e0e0e0"}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": "#555"},
            "bar": {"color": bar_color},
            "bgcolor": "#0a0e17",
            "borderwidth": 0,
            "steps": [
                {"range": [min_val, yellow_thresh], "color": "rgba(239,68,68,0.15)"},
                {"range": [yellow_thresh, green_thresh], "color": "rgba(245,158,11,0.15)"},
                {"range": [green_thresh, max_val], "color": "rgba(16,185,129,0.15)"},
            ],
        },
        number={"suffix": "", "font": {"size": 22, "color": "#e0e0e0"}},
    ))
    fig.update_layout(
        paper_bgcolor=PLOTLY_PAPER,
        plot_bgcolor=PLOTLY_BG,
        height=220,
        margin=dict(t=40, b=10, l=20, r=20),
    )
    return fig


def render(df: pd.DataFrame, models: dict = None, **kwargs) -> None:
    hide_header = kwargs.get("hide_header", False)
    if not hide_header:
        st.markdown(
            '<div class="hero-header"><div class="hero-title">🔮 Predict Tool</div>'
            '<div class="hero-subtitle">Enter a game configuration and get simultaneous ML predictions: value score, price tier, and fair-price verdict.</div></div>',
            unsafe_allow_html=True,
        )

    # ── Genre list from dataset ────────────────────────────────────────────────
    top_genres = [
        "Action", "Adventure", "Casual", "Indie", "RPG",
        "Simulation", "Sports", "Strategy", "Racing", "Free To Play"
    ]
    if df is not None and "primary_genre" in df.columns:
        genre_counts = df["primary_genre"].value_counts()
        top_genres = genre_counts.head(10).index.tolist()

    # ── Input Form ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Game Configuration</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            price = st.number_input(
                "💰 Price (USD)", min_value=0.0, max_value=200.0, value=14.99, step=0.99,
                help="The retail price you plan to set."
            )
            genre = st.selectbox(
                "🎮 Genre", options=top_genres, index=0,
                help="Primary genre of the game."
            )
            dlc_count = st.number_input(
                "📦 DLC Count", min_value=0, max_value=100, value=0,
                help="Number of DLC packs planned."
            )

        with col2:
            metacritic_score = st.slider(
                "⭐ Metacritic Score", min_value=0, max_value=100, value=0,
                help="Expected metacritic score. 0 = no score."
            )
            review_score_pct = st.slider(
                "👍 Review Score (%)", min_value=0, max_value=100, value=75,
                help="Expected positive review percentage."
            )
            platform_count = st.selectbox(
                "💻 Platform Count", options=[1, 2, 3], index=0,
                help="Number of platforms: Windows only=1, +Mac=2, +Linux=3."
            )

        with col3:
            game_age_years = st.number_input(
                "📅 Game Age (years)", min_value=0.0, max_value=30.0, value=0.5, step=0.5,
                help="How old the game is (0.5 = just launched)."
            )
            languages_count = st.number_input(
                "🌍 Languages Supported", min_value=1, max_value=40, value=5,
                help="Number of supported text languages."
            )
            is_indie = st.checkbox("🛠️ Indie Game", value=True)
            is_single_player = st.checkbox("🎯 Singleplayer", value=True)

        submitted = st.form_submit_button("🚀 Analyze Game", use_container_width=True)

    # ── Predictions ────────────────────────────────────────────────────────────
    if submitted:
        # Build game profile dict
        is_casual = 1 if "Casual" in genre else 0
        is_indie_flag = 1 if is_indie else 0
        cat_sp = 1 if is_single_player else 0
        log_reviews = float(np.log1p(max(0, review_score_pct * 10)))  # proxy for review count

        game_profile = {
            "age_by_years":               float(game_age_years),
            "categories_count":           int(platform_count) + 2,
            "languages_count":            int(languages_count),
            "peak_ccu":                   0.0,
            "log_reviews":                log_reviews,
            "genre_casual":               is_casual,
            "genre_count":                1,
            "full_audio_languages_count": min(int(languages_count) // 3, 5),
            "is_indie":                   is_indie_flag,
            "average_playtime_forever":   120.0,  # 2h default for new game
            "cat_single_player":          cat_sp,
        }

        with st.spinner("Running 3 models simultaneously…"):
            pred_value  = predict_value_score(game_profile)
            pred_tier, proba_tier = predict_price_tier(game_profile)
            fair_label, fair_conf = predict_fair_price(game_profile, price)
            pred_sweet  = predict_price_sweetspot(game_profile)
            pred_review = predict_review_score(game_profile, price)
            pred_owners = predict_ownership(game_profile, price)

        st.markdown("---")
        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

        res_col1, res_col2, res_col3 = st.columns(3)

        # ── Column 1: Value Score Gauge ────────────────────────────────────────
        with res_col1:
            st.markdown("**📊 Predicted Value Score**")
            val_clamped = max(0.0, min(pred_value, 50.0))
            fig_gauge = _gauge(
                value=round(val_clamped, 2),
                title="Quality Pts per $1",
                min_val=0.0, max_val=50.0,
                green_thresh=10.0, yellow_thresh=5.0,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            if pred_value >= 10.0:
                st.success(f"🟢 **Strong value** — {pred_value:.1f} quality pts per dollar")
            elif pred_value >= 5.0:
                st.warning(f"🟡 **Moderate value** — {pred_value:.1f} quality pts per dollar")
            else:
                st.error(f"🔴 **Low value** — {pred_value:.1f} quality pts per dollar")

        # ── Column 2: Price Tier ───────────────────────────────────────────────
        with res_col2:
            st.markdown("**🏷️ Predicted Price Tier**")
            tier_colors = {
                "Budget": "#10B981", "Mid-range": "#3B82F6",
                "Premium": "#8B5CF6", "AAA": "#F59E0B"
            }
            tier_color = tier_colors.get(pred_tier, "#e0e0e0")
            st.markdown(
                f'<div style="text-align:center; padding: 16px; border: 2px solid {tier_color}; '
                f'border-radius: 8px; background: rgba(0,0,0,0.3); margin:8px 0;">'
                f'<div style="font-size:2.2rem;">'
                f'{"💰" if pred_tier=="Budget" else "💳" if pred_tier=="Mid-range" else "💎" if pred_tier=="Premium" else "👑"}'
                f'</div>'
                f'<div style="font-size:1.5rem; font-weight:700; color:{tier_color}; margin:6px 0;">'
                f'{pred_tier}</div>'
                f'<div style="font-size:0.9rem; color:#888;">Sweetspot: ${pred_sweet:.2f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            # Tier confidence bar
            if proba_tier:
                top_conf = proba_tier.get(pred_tier, 0.0)
                st.markdown(f"**Confidence:** {top_conf*100:.1f}%")
                for t, p in sorted(proba_tier.items(), key=lambda x: -x[1]):
                    if p > 0.01:
                        st.progress(float(p), text=f"{t}: {p*100:.1f}%")

        # ── Column 3: Fair Price Verdict ───────────────────────────────────────
        with res_col3:
            st.markdown("**⚖️ Fair Price Verdict**")
            if fair_label == "Fair":
                st.markdown(
                    f'<div style="text-align:center; padding:20px; border: 2px solid #10B981; '
                    f'border-radius:8px; background: rgba(16,185,129,0.08); margin:8px 0;">'
                    f'<div style="font-size:3rem;">✅</div>'
                    f'<div style="font-size:1.4rem; font-weight:700; color:#10B981;">FAIR PRICE</div>'
                    f'<div style="font-size:0.85rem; color:#888; margin-top:6px;">'
                    f'Confidence: {fair_conf*100:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="text-align:center; padding:20px; border: 2px solid #EF4444; '
                    f'border-radius:8px; background: rgba(239,68,68,0.08); margin:8px 0;">'
                    f'<div style="font-size:3rem;">❌</div>'
                    f'<div style="font-size:1.4rem; font-weight:700; color:#EF4444;">OVERPRICED</div>'
                    f'<div style="font-size:0.85rem; color:#888; margin-top:6px;">'
                    f'Confidence: {fair_conf*100:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Secondary Metrics Row ──────────────────────────────────────────────
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Review Score", f"{max(0, min(pred_review, 100)):.1f}%")
        m2.metric("Estimated Owners", f"{pred_owners:,.0f}")
        m3.metric("Suggested Sweet Spot Price", f"${pred_sweet:.2f}")

        # ── Explanation Expander ───────────────────────────────────────────────
        with st.expander("📖 How was this calculated?"):
            st.markdown(f"""
**Model 1 — Value Score** (`HistGradientBoostingRegressor`, Test R²=0.16)
- Target: `quality_score / price` — measures quality points per dollar spent
- Key features: `log_reviews`, `age_by_years`, `peak_ccu`, `languages_count`
- Value = `{pred_value:.2f}` means the model predicts this game delivers **{pred_value:.1f} quality points per $1**

**Model 2 — Price Tier** (`HistGradientBoostingClassifier`, Test Accuracy=80.5%)
- Target: Categorizes a game into Budget / Mid-range / Premium / AAA
- Predicted tier: **{pred_tier}** with {proba_tier.get(pred_tier, 0)*100:.1f}% confidence
- The classifier was trained on {74000:,}+ commercially released Steam games with proper 80/20 train/test split

**Model 3 — Fair Price Verdict** (`DecisionTreeClassifier`, Test Accuracy=94.6%)
- Verdict: **{fair_label}** (confidence {fair_conf*100:.1f}%)
- Definition: A game is *fair* if its price is ≤ its genre median price, OR its value_score is ≥ its genre median value_score
- This prevents penalizing genuinely high-value premium games

**Value Score meaning:** The higher the score, the more quality (positive review %) you get per dollar. 
A score of 10 means 10 quality points per dollar — equivalent to a fully-reviewed Indie at $10.
A score < 3 signals players may feel the game is overpriced relative to its reception.
            """)
