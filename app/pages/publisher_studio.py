"""
publisher_studio.py -- Indie Developer Studio & What-If Scenario Simulator.

Features:
  - Interactive dual-scenario engine (Base vs Alternative)
  - Live ML model predictions with delta comparison
  - Strict reliance on historical benchmarks
  - Market position quadrant analysis
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.benchmarks import percentile_profile
from src.config import (
    ACCENT_COLORS,
    PRIMARY_GENRES,
    PLOTLY_BG_COLOR,
    PLOTLY_PAPER_BG,
    PRICE_TIER_LABELS,
)
from src.model_loader import (
    predict_price_tier, predict_value_score, predict_price_sweetspot,
    predict_review_score, predict_ownership
)
from src.similarity import find_similar_games
from src.market_position import compute_market_gap_signal, game_market_position

TIER_COLORS = {
    "Free":      "#6B7280",
    "Budget":    "#10B981",
    "Mid-range": "#3B82F6",
    "Premium":   "#8B5CF6",
    "AAA":       "#F59E0B",
}

TIER_RANGES = {
    "Free":      (0.0, 0.0),
    "Budget":    (0.01, 9.99),
    "Mid-range": (10.00, 29.99),
    "Premium":   (30.00, 59.99),
    "AAA":       (60.00, 999.00),
}

PRESETS = {
    "Custom Configuration": None,
    "Indie Action / Roguelike": {
        "name": "Chrono Blade: Genesis",
        "genre": "Action",
        "price": 14.99,
        "review_pct": 82,
        "platforms": ["Windows", "Mac", "Linux"],
        "languages": 8,
        "audio_lang": 1,
        "categories": 6,
        "is_single": True,
        "is_indie": True,
        "is_casual": False,
        "genre_count": 2,
        "peak_ccu": 1200,
        "total_rev": 650,
        "playtime": 360,
        "age_years": 1.0,
    },
    "Cozy Narrative Adventure": {
        "name": "Starlight Whispers",
        "genre": "Adventure",
        "price": 19.99,
        "review_pct": 88,
        "platforms": ["Windows", "Mac"],
        "languages": 10,
        "audio_lang": 2,
        "categories": 5,
        "is_single": True,
        "is_indie": True,
        "is_casual": False,
        "genre_count": 2,
        "peak_ccu": 450,
        "total_rev": 420,
        "playtime": 420,
        "age_years": 0.8,
    },
    "Budget Casual Arcade": {
        "name": "Neon Bounce 3D",
        "genre": "Casual",
        "price": 4.99,
        "review_pct": 72,
        "platforms": ["Windows"],
        "languages": 4,
        "audio_lang": 0,
        "categories": 3,
        "is_single": True,
        "is_indie": True,
        "is_casual": True,
        "genre_count": 2,
        "peak_ccu": 95,
        "total_rev": 120,
        "playtime": 90,
        "age_years": 0.5,
    },
    "AAA Open World RPG": {
        "name": "Elden Realms",
        "genre": "RPG",
        "price": 59.99,
        "review_pct": 92,
        "platforms": ["Windows"],
        "languages": 12,
        "audio_lang": 5,
        "categories": 8,
        "is_single": True,
        "is_indie": False,
        "is_casual": False,
        "genre_count": 3,
        "peak_ccu": 150000,
        "total_rev": 120000,
        "playtime": 4500,
        "age_years": 0.2,
    },
    "Multiplayer Co-op Shooter": {
        "name": "Strike Force: Elite",
        "genre": "Action",
        "price": 29.99,
        "review_pct": 75,
        "platforms": ["Windows", "Mac"],
        "languages": 15,
        "audio_lang": 8,
        "categories": 7,
        "is_single": False,
        "is_indie": False,
        "is_casual": False,
        "genre_count": 2,
        "peak_ccu": 45000,
        "total_rev": 15000,
        "playtime": 1200,
        "age_years": 0.1,
    },
    "Hardcore Strategy Simulator": {
        "name": "Stellar Conquest",
        "genre": "Strategy",
        "price": 39.99,
        "review_pct": 89,
        "platforms": ["Windows", "Linux"],
        "languages": 8,
        "audio_lang": 1,
        "categories": 4,
        "is_single": True,
        "is_indie": True,
        "is_casual": False,
        "genre_count": 2,
        "peak_ccu": 8500,
        "total_rev": 4500,
        "playtime": 3500,
        "age_years": 1.0,
    }
}

def _render_dual_radar(base_vals: dict, alt_vals: dict, df: pd.DataFrame, genre: str) -> go.Figure:
    """Generate a market-normalized radar chart comparing Base vs Alt vs Genre Median."""
    metrics = [
        ("price", "Price"),
        ("review_score_pct", "Review %"),
        ("recommendations", "Reviews"),
        ("peak_ccu", "Peak CCU"),
        ("average_playtime_forever", "Playtime"),
        ("languages_count", "Languages"),
    ]
    cols = [m[0] for m in metrics]
    labels = [m[1] for m in metrics]

    genre_df = df[df["primary_genre"] == genre]
    genre_med = genre_df[cols].median() if len(genre_df) > 0 else df[cols].median()
    all_med = df[cols].median()

    base_norm, alt_norm, genre_norm = [], [], []

    for col in cols:
        baseline = float(all_med.get(col, 1.0))
        if baseline <= 0 or np.isnan(baseline):
            baseline = 1.0
            
        b_val = float(base_vals.get(col, 0) or 0)
        a_val = float(alt_vals.get(col, 0) or 0)
        gen_val = float(genre_med.get(col, 0) or 0) if not pd.isna(genre_med.get(col, 0)) else 0.0

        # Scale relative to market median, capped at 4.0x
        base_norm.append(min(round(b_val / baseline, 2), 4.0))
        alt_norm.append(min(round(a_val / baseline, 2), 4.0))
        genre_norm.append(min(round(gen_val / baseline, 2), 4.0))

    fig = go.Figure()
    
    # Genre Median
    fig.add_trace(go.Scatterpolar(
        r=genre_norm + [genre_norm[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name=f"{genre} Median",
        line_color="#4B5563",
        fillcolor="rgba(75, 85, 99, 0.2)",
    ))
    
    # Base
    fig.add_trace(go.Scatterpolar(
        r=base_norm + [base_norm[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name="Base Scenario",
        line_color=ACCENT_COLORS[0],
        fillcolor="rgba(124, 106, 247, 0.25)",
    ))
    
    # Alt
    fig.add_trace(go.Scatterpolar(
        r=alt_norm + [alt_norm[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name="What-If Alternative",
        line_color=ACCENT_COLORS[1],
        fillcolor="rgba(79, 142, 247, 0.4)",
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 4.2], ticksuffix="x"),
            bgcolor="rgba(22, 27, 34, 0.6)",
        ),
        template="plotly_dark",
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_BG_COLOR,
        font=dict(color="#e0e0e0", family="Inter, sans-serif"),
        title=f"Market-Normalized Benchmark (1.0x = Global Steam Median)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=60, b=50, l=40, r=40),
    )
    return fig


def render(df: pd.DataFrame, models: dict) -> None:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🚀 Publisher Studio & What-If Engine</div>
        <div class="hero-subtitle">
            Interactive ML-driven scenario simulator. Historically benchmark your concept, 
            run isolated What-If sensitivity tests, and observe empirical market shifts.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Preset Loader ──
    preset_choice = st.selectbox("Load Predefined Scenario Template:", list(PRESETS.keys()), key="studio_preset_select")
    preset = PRESETS.get(preset_choice)
    
    # ── High Precision Models ──
    st.info("⚡ **Precision Models Active:** Predictions are powered by offline-trained HistGradientBoosting regressors.")

    # ── Dual Input UI ──
    st.markdown('<div class="section-header">Game Concept Inputs</div>', unsafe_allow_html=True)
    col_base, col_alt = st.columns(2)
    
    with col_base:
        st.markdown("**Base Scenario**", unsafe_allow_html=True)
        with st.container(border=True):
            b_name = st.text_input("Game Title", preset["name"] if preset else "Project Odyssey", key="b_name")
            g_idx = PRIMARY_GENRES.index(preset["genre"]) if preset and preset["genre"] in PRIMARY_GENRES else 0
            b_genre = st.selectbox("Primary Genre", PRIMARY_GENRES, index=g_idx, key="b_genre")
            
            c1, c2 = st.columns(2)
            with c1: b_price = st.number_input("Base Price ($)", 0.0, 99.99, float(preset["price"]) if preset else 14.99, 0.50, key="b_price", help="The target retail price in USD.")
            with c2: b_rev = st.slider("Expected Quality (%)", 10, 100, int(preset["review_pct"]) if preset else 78, key="b_rev", help="Expected percentage of positive Steam reviews (approximates quality).")
            
            c3, c4 = st.columns(2)
            with c3: b_plat = st.slider("Platforms (Count)", 1, 3, len(preset["platforms"]) if preset else 1, key="b_plat", help="Number of OS platforms supported (Windows, Mac, Linux).")
            with c4: b_lang = st.slider("Languages (Count)", 1, 30, int(preset["languages"]) if preset else 6, key="b_lang", help="Number of localized text languages supported.")
            
            b_playtime = st.number_input("Avg Playtime (min)", 0, 10000, int(preset["playtime"]) if preset else 360, 60, key="b_play", help="Expected average playtime across all users, in minutes.")
            b_total_rev = st.number_input("Expected Steam Reviews", 0, 500000, int(preset["total_rev"]) if preset else 350, 50, key="b_trev", help="Expected total volume of Steam reviews (proxy for sales volume).")
            b_peak_ccu = st.number_input("Expected Peak CCU", 0, 1000000, int(preset["peak_ccu"]) if preset else 600, 50, key="b_ccu", help="Expected peak Concurrent Users (how many playing at the same time).")

    with col_alt:
        st.markdown("**Alternative Scenario (What-If)**", unsafe_allow_html=True)
        with st.container(border=True):
            enable_alt = st.checkbox("Enable Alternative Scenario", value=True, key="enable_alt")
            
            st.text_input("Game Title", f"{b_name} (Alt)", key="a_name", disabled=True)
            a_genre = st.selectbox("Primary Genre", PRIMARY_GENRES, index=PRIMARY_GENRES.index(b_genre), key="a_genre", disabled=not enable_alt)
            
            c1, c2 = st.columns(2)
            with c1: a_price = st.number_input("Alt Price ($)", 0.0, 99.99, b_price, 0.50, key="a_price", disabled=not enable_alt)
            with c2: a_rev = st.slider("Alt Quality (%)", 10, 100, b_rev, key="a_rev", disabled=not enable_alt)
            
            c3, c4 = st.columns(2)
            with c3: a_plat = st.slider("Alt Platforms", 1, 3, b_plat, key="a_plat", disabled=not enable_alt)
            with c4: a_lang = st.slider("Alt Languages", 1, 30, b_lang, key="a_lang", disabled=not enable_alt)
            
            a_playtime = st.number_input("Alt Playtime (min)", 0, 10000, b_playtime, 60, key="a_play", disabled=not enable_alt)
            a_total_rev = st.number_input("Alt Reviews", 0, 500000, b_total_rev, 50, key="a_trev", disabled=not enable_alt)
            a_peak_ccu = st.number_input("Alt Peak CCU", 0, 1000000, b_peak_ccu, 50, key="a_ccu", disabled=not enable_alt)


    # ── Model Data Assembly ──
    def build_profile(p_price, p_rev, p_plat, p_lang, p_playtime, p_trev, p_ccu):
        return {
            "quality_score": float(p_rev),
            "age_by_years": 1.0,
            "categories_count": 5.0,
            "languages_count": float(p_lang),
            "peak_ccu": float(p_ccu) if pd.notna(p_ccu) else 0.0,
            "log_reviews": float(np.log1p(max(0.0, float(p_trev)))) if pd.notna(p_trev) else 0.0,
            "genre_casual": 0.0,
            "genre_count": 2.0,
            "full_audio_languages_count": 1.0,
            "is_indie": 1.0,
            "average_playtime_forever": float(p_playtime),
            "cat_single_player": 1.0,
            
            # Additional for radar/similarity
            "price": float(p_price) if pd.notna(p_price) else 0.0,
            "review_score_pct": (float(p_rev) / 100.0) if pd.notna(p_rev) else 0.0,
            "owners_mid": float(max(p_trev * 35.0, 1000.0)) if pd.notna(p_trev) else 1000.0,
            "recommendations": float(p_trev) if pd.notna(p_trev) else 0.0,
            "patforms_count": float(p_plat),
            "total_review": float(p_trev),
        }

    b_profile = build_profile(b_price, b_rev, b_plat, b_lang, b_playtime, b_total_rev, b_peak_ccu)
    a_profile = build_profile(a_price, a_rev, a_plat, a_lang, a_playtime, a_total_rev, a_peak_ccu) if enable_alt else b_profile

    b_val_score = predict_value_score(b_profile)
    b_tier_pred, b_proba = predict_price_tier(b_profile)
    b_price_sweetspot = predict_price_sweetspot(b_profile)
    b_expected_review = predict_review_score(b_profile, b_price)
    b_expected_owners = predict_ownership(b_profile, b_price)
    
    a_val_score = predict_value_score(a_profile)
    a_tier_pred, a_proba = predict_price_tier(a_profile)
    a_price_sweetspot = predict_price_sweetspot(a_profile)
    a_expected_review = predict_review_score(a_profile, a_price)
    a_expected_owners = predict_ownership(a_profile, a_price)

    # ── Outputs ──
    st.markdown('<div class="section-header">Market Alignment & ML Predictions</div>', unsafe_allow_html=True)
    
    s1, s2 = st.columns(2)
    with s1:
        st.metric("🎯 Suggested Base Price Sweetspot", f"${b_price_sweetspot:.2f}")
        st.caption(f"⚠️ Maximum Recommended Ceiling: **${b_price_sweetspot * 1.15:.2f}**")
    with s2:
        if enable_alt:
            delta_price = a_price_sweetspot - b_price_sweetspot
            st.metric("🎯 Suggested Alt Price Sweetspot", f"${a_price_sweetspot:.2f}", delta=f"${delta_price:.2f}")
            st.caption(f"⚠️ Alt Maximum Ceiling: **${a_price_sweetspot * 1.15:.2f}**")
            
    st.markdown("---")
    b_actual_val = (float(b_rev) / float(b_price)) if float(b_price) > 0 else float('inf')
    if enable_alt:
        a_actual_val = (float(a_rev) / float(a_price)) if float(a_price) > 0 else float('inf')
    else:
        a_actual_val = b_actual_val

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Expected Review Score", f"{b_expected_review:.1f}%")
    with k2:
        if enable_alt:
            delta_rev = a_expected_review - b_expected_review
            st.metric("Alt Expected Review", f"{a_expected_review:.1f}%", delta=f"{delta_rev:.1f}%")
        else:
            st.metric("Alt Expected Review", "-")
    with k3:
        st.metric("Expected Ownership Reach", f"{int(b_expected_owners):,}")
    with k4:
        if enable_alt:
            delta_own = a_expected_owners - b_expected_owners
            st.metric("Alt Expected Reach", f"{int(a_expected_owners):,}", delta=f"{int(delta_own):,}")
        else:
            st.metric("Alt Expected Reach", "-")

    st.markdown("---")
    t1, t2 = st.columns(2)
    with t1:
        st.metric("Base Price Tier (ML)", f"{b_tier_pred}")
    with t2:
        st.metric("Alternative Price Tier (ML)", f"{a_tier_pred}", delta="Shifted" if (enable_alt and a_tier_pred != b_tier_pred) else None)

    st.markdown("""
    <div class="info-box">
        <strong>⚠️ Note on ML Output:</strong> 
        The model outputs above reflect historical associations between product configurations and their market positioning. 
        Adjusting variables like 'Languages' or 'Price' calculates how similar configurations historically placed in the market; 
        it does <em>not</em> represent guaranteed future revenue or unit sales multipliers.
    </div>
    """, unsafe_allow_html=True)

    tab_radar, tab_pos, tab_sim = st.tabs(["📊 Radar & Benchmarks", "📉 Market Position", "🔍 Comparable Games"])

    with tab_radar:
        c1, c2 = st.columns([1.1, 0.9])
        with c1:
            fig_radar = _render_dual_radar(b_profile, a_profile, df, b_genre)
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with col_alt:
            pass
            
        with c2:
            st.markdown(f"**Percentile Shifts ({b_genre})**")
            b_pct = percentile_profile(b_profile, df, genre=b_genre)
            a_pct = percentile_profile(a_profile, df, genre=a_genre)
            
            if not b_pct.empty and not a_pct.empty:
                comp_df = b_pct[["display", "genre_pct"]].copy()
                comp_df.columns = ["Metric", "Base %ile"]
                comp_df["Alt %ile"] = a_pct["genre_pct"].values
                
                comp_df["Base %ile"] = comp_df["Base %ile"].apply(lambda x: f"{x:.1f}%")
                if enable_alt:
                    comp_df["Alt %ile"] = comp_df["Alt %ile"].apply(lambda x: f"{x:.1f}%")
                else:
                    comp_df["Alt %ile"] = "-"
                    
                st.dataframe(comp_df, hide_index=True, use_container_width=True)

    with tab_pos:
        # Market Position outputs
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Base Market Segment**")
            b_pos = game_market_position(b_profile, df)
            st.info(f"Price x Quality: **{b_pos['price_quality_pos']}**")
            st.write(f"Target Price ${b_price:.2f} vs Market Median ${b_pos['price_threshold']:.2f}")
            st.write(f"Target Quality {b_rev}% vs Market Median {b_pos['quality_threshold']:.1f}%")
            
        with c2:
            if enable_alt:
                st.markdown("**Alternative Market Segment**")
                a_pos = game_market_position(a_profile, df)
                st.info(f"Price x Quality: **{a_pos['price_quality_pos']}**")
                st.write(f"Alt Price ${a_price:.2f} vs Market Median ${a_pos['price_threshold']:.2f}")
                st.write(f"Alt Quality {a_rev}% vs Market Median {a_pos['quality_threshold']:.1f}%")
        
        st.markdown("---")
        # Gap Signal
        st.markdown("**Historical Market Gap Signal**")
        gap_df = compute_market_gap_signal(df)
        b_gap = gap_df[(gap_df["primary_genre"] == b_genre) & (gap_df["price_tier"] == b_tier_pred)]
        if not b_gap.empty:
            g_val = float(b_gap['gap_signal_norm'].iloc[0])
            st.metric(f"Gap Signal for {b_genre} / {b_tier_pred}", f"{g_val:.1f} / 100")
            st.caption("A higher signal suggests historically higher observed demand relative to existing supply in this segment.")
        else:
            st.write("Insufficient data to compute market gap for this specific segment.")

    with tab_sim:
        st.markdown("**Empirically Similar Published Titles (Base Scenario)**")
        b_profile["name"] = b_name
        b_profile["primary_genre"] = b_genre
        try:
            similar_df = find_similar_games(b_profile, df, n=15)
            if not similar_df.empty:
                display_cols = [
                    "name", "price", "review_score_pct", "peak_ccu",
                    "average_playtime_forever", "primary_genre",
                    "price_tier", "similarity_score",
                ]
                avail = [c for c in display_cols if c in similar_df.columns]
                renamed = similar_df[avail].copy()
                renamed["price"] = renamed["price"].apply(lambda p: f"${p:.2f}")
                renamed["review_score_pct"] = renamed["review_score_pct"].apply(lambda r: f"{r*100:.1f}%")
                renamed["similarity_score"] = renamed["similarity_score"].apply(lambda s: f"{s*100:.1f}%")

                column_labels = {
                    "name": "Game Title",
                    "price": "Price",
                    "review_score_pct": "Review %",
                    "peak_ccu": "Peak CCU",
                    "average_playtime_forever": "Playtime (min)",
                    "primary_genre": "Genre",
                    "price_tier": "Tier",
                    "similarity_score": "Cosine Similarity",
                }
                renamed.rename(columns=column_labels, inplace=True)
                st.dataframe(renamed, use_container_width=True, hide_index=True)
            else:
                st.info("No matching similar titles found.")
        except Exception as e:
            st.warning(f"Unable to retrieve similar titles: {e}")

