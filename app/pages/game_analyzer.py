"""
Game Analyzer page — Individual game analytical profile with model predictions,
benchmark comparisons, similar games, and market position.
"""
from __future__ import annotations

import numpy as np
import streamlit as st
import pandas as pd

from src.benchmarks import compute_genre_benchmarks, benchmark_game
from src.similarity import find_similar_games, find_similar_by_appid
from src.market_position import game_market_position
from src.model_loader import predict_value_score, predict_price_tier
from src.feature_engineering import build_model_input
import src.visualization as vz


def _game_to_model_profile(game_row: pd.Series, df: pd.DataFrame) -> dict:
    """Convert a dataset game row to the model feature dict."""
    row = game_row.to_dict()
    
    def _get(k):
        v = row.get(k)
        return 0.0 if pd.isna(v) or v is None else float(v)

    return {
        "quality_score":              _get("review_score_pct") * 100,
        "age_by_years":               _get("age_by_years"),
        "categories_count":           _get("categories_count"),
        "languages_count":            _get("languages_count"),
        "peak_ccu":                   _get("peak_ccu"),
        "log_reviews":                float(np.log1p(_get("total_review"))),
        "genre_casual":               _get("genre_casual"),
        "genre_count":                _get("genre_count"),
        "full_audio_languages_count": _get("full_audio_languages_count"),
        "is_indie":                   _get("is_indie"),
        "average_playtime_forever":   _get("average_playtime_forever"),
        "cat_single_player":          _get("cat_single_player"),
    }


def render(df: pd.DataFrame, models: dict) -> None:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🎮 Game Analyzer</div>
        <div class="hero-subtitle">
            Analytical profile for any Steam game — benchmark comparisons, market position, similar titles, and ML predictions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Select mode",
        ["Search existing game", "Analyze custom game profile"],
        horizontal=True,
    )

    # ── Mode A: Search existing game ────────────────────────────────────────────
    if mode == "Search existing game":
        popular_games = df.sort_values("total_review", ascending=False)["name"].dropna().head(15000).tolist()
        game_name = st.selectbox("Search game name (Top 15k most reviewed)", options=popular_games, index=None, placeholder="e.g. Hollow Knight, Stardew Valley…")
        game_row = None

        if game_name:
            matches = df[df["name"] == game_name]
            if not matches.empty:
                game_row = matches.iloc[0]
                _render_game_profile(game_row, df)

    # ── Mode B: Custom profile ────────────────────────────────────────────────
    else:
        st.markdown('<div class="section-header">Enter Game Profile</div>', unsafe_allow_html=True)
        with st.form("custom_game_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                name       = st.text_input("Game Name",              value="My Indie Game")
                price      = st.number_input("Price ($)",             min_value=0.0, max_value=200.0, value=14.99)
                quality    = st.slider("Review Score (%)",            0, 100, 75)
                age        = st.number_input("Game Age (years)",      min_value=0.0, max_value=30.0, value=1.0)
            with col2:
                languages  = st.number_input("Languages Supported",   min_value=1, max_value=100, value=5)
                categories = st.number_input("Steam Categories Count",min_value=1, max_value=30, value=4)
                genres     = st.number_input("Genre Count",           min_value=1, max_value=10, value=2)
                audio_lang = st.number_input("Full Audio Languages",  min_value=0, max_value=20, value=1)
            with col3:
                peak_ccu   = st.number_input("Peak CCU",              min_value=0, value=500)
                total_rev  = st.number_input("Total Reviews",         min_value=0, value=200)
                playtime   = st.number_input("Avg Playtime (minutes)", min_value=0, value=360)
                is_indie   = st.checkbox("Indie Game", value=True)
                cat_sp     = st.checkbox("Single-Player", value=True)
                genre      = st.selectbox("Primary Genre",
                    ["Action","Adventure","Casual","Indie","RPG","Simulation","Strategy","Racing","Sports","Other"])

            submitted = st.form_submit_button("Analyze")

        if submitted:
            profile = {
                "quality_score":              float(quality),
                "age_by_years":               float(age),
                "categories_count":           float(categories),
                "languages_count":            float(languages),
                "peak_ccu":                   float(peak_ccu),
                "log_reviews":                float(np.log1p(total_rev)),
                "genre_casual":               1.0 if genre == "Casual" else 0.0,
                "genre_count":                float(genres),
                "full_audio_languages_count": float(audio_lang),
                "is_indie":                   1.0 if is_indie else 0.0,
                "average_playtime_forever":   float(playtime),
                "cat_single_player":          1.0 if cat_sp else 0.0,
            }
            # Create a synthetic game_row for benchmarking
            game_values = {
                "name":             name,
                "price":            price,
                "review_score_pct": quality / 100,
                "owners_mid":       None,
                "recommendations":  None,
                "peak_ccu":         peak_ccu,
                "average_playtime_forever": playtime,
                "languages_count":  languages,
                "patforms_count":   1.0,
                "total_review":     total_rev,
                "age_by_years":     age,
                "primary_genre":    genre,
            }
            _render_model_results(profile, game_values, df)


def _render_game_profile(game_row: pd.Series, df: pd.DataFrame) -> None:
    """Full analytical profile for an existing dataset game."""
    row = game_row.to_dict()

    # ── Game header ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:1.5rem; font-weight:700; color:#e0e0e0;">{row.get('name', '')}</div>
        <div style="color:#8b949e; margin-top:6px;">
            Genre: {row.get('primary_genre', 'Unknown')} · 
            Price: ${row.get('price', 0):.2f} · 
            Tier: {row.get('price_tier', 'Unknown')} · 
            Released: {row.get('release_date', 'Unknown')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key stats ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    review_pct = row.get("review_score_pct")
    with c1: st.metric("Genre",         f"{row.get('primary_genre', 'Unknown')}")
    with c2: st.metric("Price",         f"${row.get('price', 0):.2f}" if pd.notna(row.get('price')) else "Not available")
    with c3: st.metric("Review Score",  f"{review_pct*100:.1f}%" if pd.notna(review_pct) else "Not available")
    with c4: st.metric("Est. Owners",   f"{int(row.get('owners_mid',0)):,}" if pd.notna(row.get('owners_mid')) else "Not available")
    with c5: st.metric("Release Year",  f"{int(row.get('release_year', 0))}" if pd.notna(row.get('release_year')) else "Not available")

    # ── Model predictions ─────────────────────────────────────────────────────
    model_profile = _game_to_model_profile(game_row, df)
    _render_model_results(model_profile, row, df, game_row=game_row)


def _render_model_results(model_profile: dict, game_values: dict, df: pd.DataFrame,
                          game_row: pd.Series | None = None) -> None:
    """Render model predictions + benchmark + similar games."""

    val_pred = None
    tier_pred = None
    try:
        val_pred = predict_value_score(model_profile)
        tier_pred, _ = predict_price_tier(model_profile)
    except:
        pass

    # ── Executive Report Download ─────────────────────────────────────────────
    owners = game_values.get('owners_mid')
    if owners is None or pd.isna(owners): owners = 0
    
    peak = game_values.get('peak_ccu')
    if peak is None or pd.isna(peak): peak = 0
    
    price = game_values.get('price')
    if price is None or pd.isna(price): price = 0.0
    
    review = game_values.get('review_score_pct')
    if review is None or pd.isna(review): review = 0.0

    st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)
    report_md = f"""# Executive Report: {game_values.get('name', 'Unknown Game')}

## 1. Game Profile
- **Primary Genre**: {game_values.get('primary_genre', 'Unknown')}
- **Price**: ${price:.2f}
- **Review Score**: {(review * 100):.1f}%
- **Estimated Owners**: {int(owners):,}
- **Peak CCU**: {int(peak):,}

## 2. ML Predictions
- **Predicted Value Score**: {f"{val_pred:.2f} pts/$" if val_pred else 'N/A'}
- **Predicted Price Tier**: {tier_pred if tier_pred else 'N/A'}

## 3. Methodology & Caveats
All metrics are based on statistical associations from the Steam dataset. No causal claims are made regarding commercial success. This report is for analytical purposes only.
"""
    st.download_button(
        label="📥 Download Executive Report (Markdown)",
        data=report_md,
        file_name=f"executive_report_{game_values.get('name', 'game').replace(' ', '_').lower()}.md",
        mime="text/markdown",
    )


    # ── ML Predictions ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">ML Model Predictions</div>', unsafe_allow_html=True)
    try:
        val_pred  = predict_value_score(model_profile)
        tier_pred, tier_proba = predict_price_tier(model_profile)

        c1, c2 = st.columns(2)
        with c1:
            val_display = f"{val_pred:.2f} pts/$" if pd.notna(game_values.get("review_score_pct")) else "N/A (No Review Data)"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val_display}</div>
                <div class="metric-label">Predicted Value Score (quality points per $1)</div>
                <div style="font-size:0.8rem; color:#8b949e; margin-top:8px;">
                    Higher = more quality per dollar spent. Calculated as review quality / price.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            proba_str = " · ".join([f"{k}: {v*100:.0f}%" for k, v in tier_proba.items()])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{tier_pred}</div>
                <div class="metric-label">Predicted Natural Price Tier</div>
                <div style="font-size:0.8rem; color:#8b949e; margin-top:8px;">{proba_str}</div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Model prediction failed: {e}")

    # ── Benchmark & Percentiles ────────────────────────────────────────────────
    genre = game_values.get("primary_genre")
    genre_benchmarks = compute_genre_benchmarks(df)

    from src.benchmarks import percentile_profile
    pct_profile = percentile_profile(game_values, df, genre=genre)

    st.markdown('<div class="section-header">Market Position Percentiles & Radar Profile</div>', unsafe_allow_html=True)
    if not pct_profile.empty:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            fig = vz.percentile_profile_bar(pct_profile, f"Percentile Profile — {game_values.get('name', 'Game')}")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            radar_labels = ["Price ($)", "Review Score (%)", "Est. Owners", "Peak CCU", "Avg Playtime (min)"]
            game_pcts = {r["display"]: r["market_pct"] for _, r in pct_profile.iterrows()}
            bench_pcts = {r["display"]: 50 for _, r in pct_profile.iterrows()}
            fig_radar = vz.radar_chart(game_pcts, bench_pcts, radar_labels)
            st.plotly_chart(fig_radar, use_container_width=True)

    # ── Context Plots ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Price & Quality Context</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if genre and genre_benchmarks is not None and not genre_benchmarks.empty:
            gm = genre_benchmarks[genre_benchmarks["primary_genre"] == genre]
            if not gm.empty:
                g_med_price = float(gm.iloc[0].get("median_price", df[df["primary_genre"]==genre]["price"].median()))
                p25 = df[df["primary_genre"]==genre]["price"].quantile(0.25)
                p75 = df[df["primary_genre"]==genre]["price"].quantile(0.75)
                fig_price = vz.price_context_distribution(df, game_values.get("price"), genre, g_med_price, p25, p75)
                st.plotly_chart(fig_price, use_container_width=True)
            else:
                st.info("Not enough genre data for price context.")
    with c2:
        if genre and genre_benchmarks is not None and not genre_benchmarks.empty:
            gm = genre_benchmarks[genre_benchmarks["primary_genre"] == genre]
            if not gm.empty:
                g_med_qual = float(gm.iloc[0].get("median_review_score_pct", df[df["primary_genre"]==genre]["review_score_pct"].median()))
                fig_qual = vz.quality_context_distribution(df, game_values.get("review_score_pct"), genre, g_med_qual)
                st.plotly_chart(fig_qual, use_container_width=True)

    # ── Market Position Map ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Market Position Map</div>', unsafe_allow_html=True)
    if genre:
        fig_map = vz.market_position_scatter(df, game_values.get("price"), game_values.get("review_score_pct"), genre, game_values.get("name", "Game"))
        st.plotly_chart(fig_map, use_container_width=True)

    # ── Engagement & Global Reach ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Engagement & Global Reach</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Avg Playtime", f"{game_values.get('average_playtime_forever', 0):.0f} min" if pd.notna(game_values.get('average_playtime_forever')) else "Not available")
    with c2: st.metric("Peak CCU", f"{game_values.get('peak_ccu', 0):,}" if pd.notna(game_values.get('peak_ccu')) else "Not available")
    with c3: st.metric("Supported Languages", f"{game_values.get('languages_count', 0)}" if pd.notna(game_values.get('languages_count')) else "Not available")
    with c4: st.metric("Platforms", f"{game_values.get('patforms_count', 1)}" if pd.notna(game_values.get('patforms_count')) else "Not available")

    # ── Anomaly Detection ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Anomaly Detection</div>', unsafe_allow_html=True)
    
    # 1. AI Anomaly Detection (Isolation Forest)
    try:
        from app.pages.anomaly_finder import compute_isolation_forest_anomalies
        iso_df = compute_isolation_forest_anomalies(df, contamination=0.02)
        
        if game_row is not None and "name" in game_row:
            iso_match = iso_df[iso_df["name"] == game_row["name"]]
            if not iso_match.empty:
                score = iso_match.iloc[0]["anomaly_score"]
                st.warning(f"🤖 **AI Anomaly Flag:** The Isolation Forest model classifies this game as a statistical market anomaly (Score: **{score:.2f}**). Its combination of metrics is highly unusual compared to the broader Steam ecosystem.")
            else:
                st.success("🤖 **AI Detection:** This game's multidimensional profile falls within normal statistical boundaries (not flagged as an anomaly by Isolation Forest).")
    except Exception as e:
        pass # Fallback to percentiles if AI model fails to load

    # 2. Metric Percentile Extremes
    if not pct_profile.empty:
        anomalies = []
        for _, row in pct_profile.iterrows():
            if row["market_pct"] >= 95:
                anomalies.append(f"⬆️ **Exceptionally High {row['display']}**: Top {100 - row['market_pct']:.1f}% of market")
            elif row["market_pct"] <= 5:
                anomalies.append(f"⬇️ **Exceptionally Low {row['display']}**: Bottom {row['market_pct']:.1f}% of market")
        
        if anomalies:
            for a in anomalies:
                st.markdown(f"- {a}")
        else:
            st.info("No single metric is exceptionally high (>95th pct) or low (<5th pct).")

    # ── Similar Games ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Similar Games</div>', unsafe_allow_html=True)
    try:
        if game_row is not None and "app_id" in game_row.index:
            sim_df = find_similar_by_appid(int(game_row["app_id"]), df, n=10)
        else:
            sim_df = find_similar_games(game_values, df, n=10)

        display_cols = ["name", "similarity_score", "price", "review_score_pct",
                        "owners_mid", "primary_genre", "price_tier"]
        available = [c for c in display_cols if c in sim_df.columns]
        st.dataframe(sim_df[available].round(3), use_container_width=True, hide_index=True)

        # Similar game comparison heatmap
        metrics = ["price", "review_score_pct", "owners_mid", "average_playtime_forever",
                   "languages_count", "patforms_count"]
        available_m = [m for m in metrics if m in sim_df.columns]
        if available_m:
            fig3 = vz.similar_games_heatmap(sim_df.head(10), available_m)
            st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute similar games: {e}")
