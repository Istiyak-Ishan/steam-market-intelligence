"""
game_comparison.py -- Head-to-Head Multi-Game Comparison Engine.

Features:
  - Side-by-side comparison of 2 to 4 Steam games simultaneously
  - Curated popular showdown presets + custom search across the dataset
  - Normalised multi-game overlay radar chart
  - Clustered comparative bar charts across core metrics
  - Price vs Quality market quadrant positioning
  - Live ML Model Predictions (Value Score & Price Tier) per game
  - Comprehensive technical and commercial spec sheet with CSV download
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    ACCENT_COLORS,
    PLOTLY_BG_COLOR,
    PLOTLY_PAPER_BG,
    PRICE_TIER_LABELS,
)
from src.model_loader import predict_price_tier, predict_value_score

PRESET_SHOWDOWNS = {
    "Indie Masterpieces": ["Hollow Knight", "Celeste", "Terraria"],
    "Survival Giants": ["Rust", "The Forest", "ARK: Survival Evolved"],
    "Epic RPGs": ["The Witcher 3: Wild Hunt", "Cyberpunk 2077"],
    "Valve Classics": ["Portal 2", "Left 4 Dead 2"],
    "Custom Selection": [],
}


def _extract_model_profile(row: pd.Series | dict) -> dict:
    """Build MODEL_FEATURES dictionary from a game row."""
    if isinstance(row, pd.Series):
        row = row.to_dict()
    q_score = row.get("quality_score")
    if q_score is None or np.isnan(float(q_score)):
        rev_pct = row.get("review_score_pct", 0) or 0
        q_score = float(rev_pct) * 100.0

    return {
        "quality_score":              float(q_score),
        "age_by_years":               float(row.get("age_by_years", 0) or 0),
        "categories_count":           float(row.get("categories_count", 0) or 0),
        "languages_count":            float(row.get("languages_count", 0) or 0),
        "peak_ccu":                   float(row.get("peak_ccu", 0) or 0),
        "log_reviews":                float(row.get("log_reviews", 0) or np.log1p(row.get("total_review", 0) or 0)),
        "genre_casual":               float(row.get("genre_casual", 0) or 0),
        "genre_count":                float(row.get("genre_count", 0) or 0),
        "full_audio_languages_count": float(row.get("full_audio_languages_count", 0) or 0),
        "is_indie":                   float(row.get("is_indie", 0) or 0),
        "average_playtime_forever":   float(row.get("average_playtime_forever", 0) or 0),
        "cat_single_player":          float(row.get("cat_single_player", 0) or 0),
    }


def _render_multi_radar(games: list[dict], all_medians: pd.Series) -> go.Figure:
    """Render overlaid radar chart for up to 4 games."""
    metrics = [
        ("price", "Price"),
        ("review_score_pct", "Review %"),
        ("total_review", "Reviews"),
        ("peak_ccu", "Peak CCU"),
        ("average_playtime_forever", "Playtime"),
        ("languages_count", "Languages"),
        ("patforms_count", "Platforms"),
    ]
    cols = [m[0] for m in metrics]
    labels = [m[1] for m in metrics]

    fig = go.Figure()

    for idx, game in enumerate(games):
        norm_vals = []
        for col in cols:
            med = float(all_medians.get(col, 1.0))
            if med <= 0 or np.isnan(med):
                med = 1.0
            raw_val = float(game.get(col, 0) or 0)
            norm_vals.append(min(round(raw_val / med, 2), 4.5))

        color = ACCENT_COLORS[idx % len(ACCENT_COLORS)]
        fig.add_trace(go.Scatterpolar(
            r=norm_vals + [norm_vals[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name=game["name"],
            line_color=color,
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 4.5], ticksuffix="x"),
            bgcolor="rgba(22, 27, 34, 0.6)",
        ),
        template="plotly_dark",
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_BG_COLOR,
        font=dict(color="#e0e0e0", family="Inter, sans-serif"),
        title="Multi-Game Benchmark Radar (1.0x = Steam Market Median)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=50, b=50, l=40, r=40),
    )
    return fig


def render(df: pd.DataFrame, models: dict | None = None) -> None:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">Game Comparison Engine</div>
        <div class="hero-subtitle">
            Head-to-head multi-game analytics — evaluate competitor titles across pricing,
            player reviews, engagement, radar profiles, and algorithmic ML value scores.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Selection Controls ───────────────────────────────────────────────────
    if "comp_selected_titles" not in st.session_state:
        st.session_state.comp_selected_titles = PRESET_SHOWDOWNS["Indie Masterpieces"]

    def _on_preset_change():
        preset = st.session_state.comp_preset_select
        if preset != "Custom Selection":
            st.session_state.comp_selected_titles = PRESET_SHOWDOWNS.get(preset, [])[:4]

    with st.expander("🎮 Select Games to Compare (2 to 4 titles)", expanded=True):
        col_mode, col_search = st.columns([1, 2])

        with col_mode:
            st.selectbox(
                "Preset Showdown:",
                list(PRESET_SHOWDOWNS.keys()),
                key="comp_preset_select",
                on_change=_on_preset_change,
            )

        with col_search:
            game_names = df["name"].dropna().drop_duplicates().tolist()
            selected_names = st.multiselect(
                "Search & Select Titles for Comparison (Min 2, Max 4):",
                options=game_names,
                max_selections=4,
                placeholder="e.g. Hades, Dead Cells, Elden Ring...",
                key="comp_selected_titles",
            )

    if len(selected_names) < 2:
        st.info("💡 Please select at least **2 games** to run the head-to-head comparison.")
        return

    # Extract rows for selected titles
    selected_rows = []
    for name in selected_names:
        sub = df[df["name"] == name]
        if not sub.empty:
            selected_rows.append(sub.iloc[0])

    if len(selected_rows) < 2:
        st.warning("Could not load data for selected titles.")
        return

    # Market medians for normalisation
    radar_cols = ["price", "review_score_pct", "total_review", "peak_ccu",
                  "average_playtime_forever", "languages_count", "patforms_count"]
    market_medians = df[radar_cols].median()

    # ── Summary KPI Cards Row ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Head-to-Head Overview</div>', unsafe_allow_html=True)
    kpi_cols = st.columns(len(selected_rows))

    for idx, (col, row) in enumerate(zip(kpi_cols, selected_rows)):
        color = ACCENT_COLORS[idx % len(ACCENT_COLORS)]
        p = float(row.get("price", 0) or 0)
        q = float(row.get("review_score_pct", 0) or 0) * 100.0
        ccu = int(row.get("peak_ccu", 0) or 0)
        owners = int(row.get("owners_mid", 0) or 0)

        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {color};">
                <div style="font-size:1.15rem; font-weight:700; color:{color}; margin-bottom:6px;">{row['name']}</div>
                <div style="font-size:1.6rem; font-weight:700; color:#e0e0e0;">${p:.2f}</div>
                <div style="font-size:0.85rem; color:#8b949e; margin-bottom:8px;">{row.get('primary_genre', 'Unknown')} · {row.get('price_tier', 'N/A')}</div>
                <div style="border-top: 1px solid #30363d; padding-top:6px; font-size:0.85rem;">
                    <div>⭐ <strong>{q:.1f}%</strong> Review Score</div>
                    <div>👥 <strong>{owners:,}</strong> Est Owners</div>
                    <div>⚡ <strong>{ccu:,}</strong> Peak CCU</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Comparison Tabs ──────────────────────────────────────────────────────
    tab_radar, tab_bars, tab_matrix, tab_ml, tab_specs = st.tabs([
        "🕸️ Benchmark Radar",
        "📊 Head-to-Head Bars",
        "⚖️ Price-Quality Matrix",
        "🤖 ML Value Scores",
        "📋 Comprehensive Specs",
    ])

    # ── Tab 1: Benchmark Radar ───────────────────────────────────────────────
    with tab_radar:
        st.markdown('<div class="section-header">Multidimensional Overlay Radar</div>', unsafe_allow_html=True)
        radar_dict_list = [r.to_dict() for r in selected_rows]
        fig_radar = _render_multi_radar(radar_dict_list, market_medians)
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("""
        <div class="info-box" style="font-size:0.85rem;">
            <strong>Interpretation:</strong> Dimensions are normalised against the overall Steam market median (1.0x).
            Points extending past 1.0x represent performance multiples above the median Steam title.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2: Head-to-Head Bars ─────────────────────────────────────────────
    with tab_bars:
        st.markdown('<div class="section-header">Direct Metric Comparisons</div>', unsafe_allow_html=True)

        col_b1, col_b2 = st.columns(2)
        comp_df = pd.DataFrame([
            {
                "Game": r["name"],
                "Price ($)": float(r.get("price", 0) or 0),
                "Review %": float(r.get("review_score_pct", 0) or 0) * 100.0,
                "Peak CCU": float(r.get("peak_ccu", 0) or 0),
                "Avg Playtime (h)": float(r.get("average_playtime_forever", 0) or 0) / 60.0,
                "Languages": float(r.get("languages_count", 0) or 0),
                "Platforms": float(r.get("patforms_count", 0) or 0),
            }
            for r in selected_rows
        ])

        with col_b1:
            fig_p = px.bar(
                comp_df,
                x="Game",
                y="Price ($)",
                color="Game",
                text=comp_df["Price ($)"].apply(lambda p: f"${p:.2f}"),
                template="plotly_dark",
                color_discrete_sequence=ACCENT_COLORS,
                title="Launch Price Comparison",
            )
            fig_p.update_layout(paper_bgcolor=PLOTLY_PAPER_BG, plot_bgcolor=PLOTLY_BG_COLOR, showlegend=False)
            st.plotly_chart(fig_p, use_container_width=True)

            fig_ccu = px.bar(
                comp_df,
                x="Game",
                y="Peak CCU",
                color="Game",
                text=comp_df["Peak CCU"].apply(lambda c: f"{int(c):,}"),
                template="plotly_dark",
                color_discrete_sequence=ACCENT_COLORS,
                title="Peak Concurrent Players (CCU)",
            )
            fig_ccu.update_layout(paper_bgcolor=PLOTLY_PAPER_BG, plot_bgcolor=PLOTLY_BG_COLOR, showlegend=False)
            st.plotly_chart(fig_ccu, use_container_width=True)

        with col_b2:
            fig_rev = px.bar(
                comp_df,
                x="Game",
                y="Review %",
                color="Game",
                text=comp_df["Review %"].apply(lambda q: f"{q:.1f}%"),
                template="plotly_dark",
                color_discrete_sequence=ACCENT_COLORS,
                title="Review Score (% Positive)",
            )
            fig_rev.update_layout(paper_bgcolor=PLOTLY_PAPER_BG, plot_bgcolor=PLOTLY_BG_COLOR, showlegend=False, yaxis_range=[0, 105])
            st.plotly_chart(fig_rev, use_container_width=True)

            fig_pt = px.bar(
                comp_df,
                x="Game",
                y="Avg Playtime (h)",
                color="Game",
                text=comp_df["Avg Playtime (h)"].apply(lambda h: f"{h:.1f}h"),
                template="plotly_dark",
                color_discrete_sequence=ACCENT_COLORS,
                title="Average Playtime per Owner (Hours)",
            )
            fig_pt.update_layout(paper_bgcolor=PLOTLY_PAPER_BG, plot_bgcolor=PLOTLY_BG_COLOR, showlegend=False)
            st.plotly_chart(fig_pt, use_container_width=True)

    # ── Tab 3: Price-Quality Matrix ──────────────────────────────────────────
    with tab_matrix:
        st.markdown('<div class="section-header">Market Positioning Matrix (Price vs Quality)</div>', unsafe_allow_html=True)

        # Background sample for market context
        selected_genres = [r.get("primary_genre") for r in selected_rows if r.get("primary_genre")]
        bg_sample = df[df["primary_genre"].isin(selected_genres)].sample(min(800, len(df)), random_state=42)

        fig_quad = px.scatter(
            bg_sample,
            x="price",
            y="review_score_pct",
            opacity=0.25,
            template="plotly_dark",
            color_discrete_sequence=["#4B5563"],
            hover_name="name",
            labels={"price": "Price ($ USD)", "review_score_pct": "Review Score (% Positive)"},
            title="Position Relative to Market Median Crosshairs",
        )

        med_p = float(df["price"].median())
        med_q = float(df["review_score_pct"].dropna().median())

        fig_quad.add_vline(x=med_p, line_dash="dash", line_color="#9CA3AF", annotation_text=f"Market Median ${med_p:.2f}")
        fig_quad.add_hline(y=med_q, line_dash="dash", line_color="#9CA3AF", annotation_text=f"Market Median {med_q*100:.1f}%")

        # Overlay highlighted games
        for idx, r in enumerate(selected_rows):
            color = ACCENT_COLORS[idx % len(ACCENT_COLORS)]
            fig_quad.add_trace(go.Scatter(
                x=[float(r.get("price", 0) or 0)],
                y=[float(r.get("review_score_pct", 0) or 0)],
                mode="markers+text",
                marker=dict(size=16, color=color, line=dict(color="#FFFFFF", width=2)),
                text=[r["name"]],
                textposition="top center",
                textfont=dict(color="#FFFFFF", size=12, family="Inter"),
                name=r["name"],
            ))

        fig_quad.update_layout(
            paper_bgcolor=PLOTLY_PAPER_BG,
            plot_bgcolor=PLOTLY_BG_COLOR,
            font=dict(color="#e0e0e0", family="Inter, sans-serif"),
            xaxis=dict(range=[0, max([float(r.get("price", 0) or 0) for r in selected_rows] + [60]) * 1.15]),
            yaxis=dict(range=[0.4, 1.02]),
            margin=dict(t=50, b=40, l=40, r=40),
        )
        st.plotly_chart(fig_quad, use_container_width=True)

    # ── Tab 4: ML Value Scores ───────────────────────────────────────────────
    with tab_ml:
        st.markdown('<div class="section-header">Algorithmic ML Value & Pricing Predictions</div>', unsafe_allow_html=True)
        st.markdown("Random Forest Regressor & Classifier inferences for each compared title:")

        ml_rows = []
        for r in selected_rows:
            prof = _extract_model_profile(r)
            try:
                v_score = predict_value_score(prof)
                pred_tier, proba = predict_price_tier(prof)
                conf = max(proba.values()) * 100.0
            except Exception as e:
                v_score = 0.0
                pred_tier = "Error"
                conf = 0.0

            actual_p = float(r.get("price", 0) or 0)
            ml_rows.append({
                "Game Title": r["name"],
                "Actual Price": f"${actual_p:.2f}",
                "Actual Tier": r.get("price_tier", "N/A"),
                "Predicted Tier": pred_tier,
                "Tier Confidence": f"{conf:.1f}%",
                "Predicted Value Score": f"{v_score:.2f} pts/$",
                "Value Score Raw": v_score,
            })

        ml_df = pd.DataFrame(ml_rows)
        # Winner highlight
        best_val = ml_df.sort_values("Value Score Raw", ascending=False).iloc[0]["Game Title"]

        st.markdown(f"""
        <div class="info-box">
            💎 <strong>Algorithmic Value Winner:</strong> <strong>{best_val}</strong> provides the highest predicted quality points per dollar according to the Random Forest Regressor.
        </div>
        """, unsafe_allow_html=True)

        disp_cols = ["Game Title", "Actual Price", "Actual Tier", "Predicted Tier", "Tier Confidence", "Predicted Value Score"]
        st.dataframe(ml_df[disp_cols], use_container_width=True, hide_index=True)

    # ── Tab 5: Comprehensive Specs ───────────────────────────────────────────
    with tab_specs:
        st.markdown('<div class="section-header">Technical & Commercial Specification Matrix</div>', unsafe_allow_html=True)

        spec_keys = [
            ("Primary Genre", "primary_genre"),
            ("Price ($ USD)", lambda r: f"${float(r.get('price', 0) or 0):.2f}"),
            ("Price Tier", "price_tier"),
            ("Review Score", lambda r: f"{float(r.get('review_score_pct', 0) or 0)*100:.1f}%"),
            ("Total Reviews", lambda r: f"{int(r.get('total_review', 0) or 0):,}"),
            ("Peak Concurrent Players", lambda r: f"{int(r.get('peak_ccu', 0) or 0):,}"),
            ("Estimated Owners", lambda r: f"{int(r.get('owners_mid', 0) or 0):,}"),
            ("Avg Playtime (hrs)", lambda r: f"{float(r.get('average_playtime_forever', 0) or 0)/60:.1f}h"),
            ("Supported Platforms", lambda r: f"{int(r.get('patforms_count', 1) or 1)} platform(s)"),
            ("Supported Languages", lambda r: f"{int(r.get('languages_count', 1) or 1)}"),
            ("Full Audio Languages", lambda r: f"{int(r.get('full_audio_languages_count', 0) or 0)}"),
            ("Steam Feature Categories", lambda r: f"{int(r.get('categories_count', 0) or 0)}"),
            ("Single-Player", lambda r: "Yes" if r.get("cat_single_player", 0) else "No"),
            ("Indie Tag", lambda r: "Yes" if r.get("is_indie", 0) else "No"),
            ("Casual Tag", lambda r: "Yes" if r.get("genre_casual", 0) else "No"),
            ("Game Lifecycle Age", lambda r: f"{float(r.get('age_by_years', 0) or 0):.1f} years"),
        ]

        spec_matrix = {}
        spec_matrix["Specification"] = [k[0] for k in spec_keys]

        for r in selected_rows:
            col_vals = []
            for k in spec_keys:
                extractor = k[1]
                if callable(extractor):
                    val = extractor(r)
                else:
                    val = str(r.get(extractor, "—"))
                col_vals.append(val)
            spec_matrix[r["name"]] = col_vals

        spec_df = pd.DataFrame(spec_matrix)
        st.dataframe(spec_df, use_container_width=True, hide_index=True)

        # Download CSV
        csv_buff = io.StringIO()
        spec_df.to_csv(csv_buff, index=False)
        st.download_button(
            label="📥 Download Comparison Spec Sheet CSV",
            data=csv_buff.getvalue(),
            file_name=f"game_comparison_{'_vs_'.join([r['name'][:8].lower().replace(' ', '') for r in selected_rows])}.csv",
            mime="text/csv",
            key="comp_download_csv",
        )
