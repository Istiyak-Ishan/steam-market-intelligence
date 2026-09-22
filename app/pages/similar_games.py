"""
similar_games.py -- Dedicated Similar Game Finder & Competitor Intelligence Engine.

Features:
  - Find top N similar titles for any existing Steam game or custom specification
  - Cosine similarity across standardised multidimensional feature space
  - Filterable by genre, price range, and minimum similarity threshold
  - Target game vs top competitor overlay radar comparison
  - Competitor positioning scatter plot
  - Price & reception gap analysis
  - Detailed competitor data table with CSV export
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
    PRIMARY_GENRES,
    PLOTLY_BG_COLOR,
    PLOTLY_PAPER_BG,
    SIMILARITY_FEATURES,
    SIMILARITY_N_DEFAULT,
)
from src.similarity import find_similar_by_appid, find_similar_games

POPULAR_TARGETS = [
    "Hollow Knight",
    "Terraria",
    "Stardew Valley",
    "Celeste",
    "The Witcher 3: Wild Hunt",
    "Cyberpunk 2077",
    "Rust",
    "The Forest",
    "Portal 2",
    "Left 4 Dead 2",
]


def _render_comparison_radar(target_row: dict, competitor_rows: list[dict], all_medians: pd.Series) -> go.Figure:
    """Render a radar chart comparing the target title against top competitors."""
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

    # Target trace
    t_vals = []
    for col in cols:
        med = float(all_medians.get(col, 1.0))
        if med <= 0 or np.isnan(med):
            med = 1.0
        v = float(target_row.get(col, 0) or 0)
        t_vals.append(min(round(v / med, 2), 4.5))

    fig.add_trace(go.Scatterpolar(
        r=t_vals + [t_vals[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name=f"★ {target_row['name']} (Target)",
        line_color="#FFFFFF",
        line_width=3,
        fillcolor="rgba(255, 255, 255, 0.2)",
    ))

    # Top competitor traces (up to 3)
    for idx, comp in enumerate(competitor_rows[:3]):
        c_vals = []
        for col in cols:
            med = float(all_medians.get(col, 1.0))
            if med <= 0 or np.isnan(med):
                med = 1.0
            v = float(comp.get(col, 0) or 0)
            c_vals.append(min(round(v / med, 2), 4.5))

        color = ACCENT_COLORS[idx % len(ACCENT_COLORS)]
        fig.add_trace(go.Scatterpolar(
            r=c_vals + [c_vals[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name=f"{comp['name']} ({comp.get('similarity_score', 0)*100:.1f}%)",
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
        title="Target vs Top 3 Closest Competitors (1.0x = Steam Market Median)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
        margin=dict(t=50, b=60, l=40, r=40),
    )
    return fig


def render(df: pd.DataFrame) -> None:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">Similar Game Finder & Competitor Intelligence</div>
        <div class="hero-subtitle">
            Multidimensional cosine similarity engine — discover direct market substitutes, analyze competitor pricing,
            and benchmark performance against empirical peer clusters.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Mode Selection ────────────────────────────────────────────────────────
    mode = st.radio(
        "Search Approach:",
        ["Search Existing Steam Title", "Custom Market Query Profile"],
        horizontal=True,
        key="sim_mode_radio",
    )

    target_profile = {}
    target_row = None

    if mode == "Search Existing Steam Title":
        with st.expander("🔍 Select Target Game", expanded=True):
            col_quick, col_search = st.columns([1, 2])

            with col_quick:
                quick_pick = st.selectbox(
                    "Quick Popular Picks:",
                    ["Custom Search Below"] + POPULAR_TARGETS,
                    key="sim_quick_pick",
                )

            with col_search:
                search_text = st.text_input(
                    "Or Search Any Steam Title:",
                    placeholder="e.g. Slay the Spire, Dead Cells, Factorio…",
                    key="sim_title_search",
                )

            # Determine query game name
            if search_text.strip():
                matches = df[df["name"].str.contains(search_text.strip(), case=False, na=False)]
                if matches.empty:
                    st.warning("No matches found for that search query.")
                    return
                selected_name = st.selectbox(
                    "Select Exact Title Match:",
                    matches["name"].drop_duplicates().head(25).tolist(),
                    key="sim_match_select",
                )
            elif quick_pick != "Custom Search Below":
                selected_name = quick_pick
            else:
                selected_name = POPULAR_TARGETS[0]

            sub = df[df["name"] == selected_name]
            if sub.empty:
                st.warning(f"Could not load data for {selected_name}.")
                return
            target_row = sub.iloc[0]
            target_profile = target_row.to_dict()

    else:
        # Custom Query Profile
        with st.expander("⚙️ Define Target Market Profile", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                q_name = st.text_input("Project Codename", "My Upcoming Title", key="sim_cust_name")
                q_genre = st.selectbox("Primary Genre", PRIMARY_GENRES, key="sim_cust_genre")
                q_price = st.slider("Target Price ($ USD)", 0.0, 79.99, 14.99, step=0.5, key="sim_cust_price")
                q_rev_pct = st.slider("Expected Review Score (%)", 20, 100, 80, step=1, key="sim_cust_rev")

            with col2:
                q_ccu = st.number_input("Target Peak CCU", 0, 1000000, 600, step=50, key="sim_cust_ccu")
                q_playtime = st.number_input("Avg Playtime (minutes)", 0, 30000, 360, step=60, key="sim_cust_play")
                q_platforms = st.slider("Supported Platforms (Win, Mac, Linux)", 1, 3, 2, key="sim_cust_plat")

            with col3:
                q_langs = st.slider("Languages Supported", 1, 35, 6, key="sim_cust_lang")
                q_reviews = st.number_input("Expected Total Reviews", 0, 200000, 350, step=50, key="sim_cust_totrev")
                q_age = st.slider("Simulated Age (years)", 0.0, 10.0, 1.0, step=0.5, key="sim_cust_age")

            target_profile = {
                "name":                     q_name,
                "primary_genre":            q_genre,
                "price":                    float(q_price),
                "review_score_pct":         float(q_rev_pct) / 100.0,
                "owners_mid":               float(max(q_reviews * 35.0, 1000.0)),
                "recommendations":          float(q_reviews),
                "peak_ccu":                 float(q_ccu),
                "average_playtime_forever": float(q_playtime),
                "languages_count":          float(q_langs),
                "patforms_count":           float(q_platforms),
                "total_review":             float(q_reviews),
                "age_by_years":             float(q_age),
            }
            target_row = pd.Series(target_profile)

    # ── Similarity Engine Filters ─────────────────────────────────────────────
    with st.expander("🎯 Filter & Threshold Tuning", expanded=False):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            top_n = st.slider("Number of Similar Games to Retrieve:", 5, 30, 10, key="sim_top_n")
        with col_f2:
            genre_filter = st.checkbox("Restrict results to same primary genre only", value=False, key="sim_genre_restrict")
        with col_f3:
            min_sim_pct = st.slider("Minimum Similarity Match (%):", 0, 95, 50, step=5, key="sim_min_pct")

    # ── Target Game Showcase Card ─────────────────────────────────────────────
    t_name = str(target_profile.get("name", "Unknown"))
    t_price = float(target_profile.get("price", 0) or 0)
    t_rev = float(target_profile.get("review_score_pct", 0) or 0) * 100.0
    t_ccu = int(target_profile.get("peak_ccu", 0) or 0)
    t_genre = str(target_profile.get("primary_genre", "Unknown"))
    t_playtime = float(target_profile.get("average_playtime_forever", 0) or 0) / 60.0

    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid #7C3AED; margin-bottom: 20px;">
        <div style="font-size: 1.3rem; font-weight: 700; color: #FFFFFF;">
            Target: <span style="color: #A78BFA;">{t_name}</span> (${t_price:.2f} · {t_genre})
        </div>
        <div style="font-size: 0.9rem; color: #9CA3AF; margin-top: 4px;">
            ⭐ Review Score: <strong>{t_rev:.1f}%</strong> &nbsp;|&nbsp;
            ⚡ Peak CCU: <strong>{t_ccu:,}</strong> &nbsp;|&nbsp;
            ⏳ Avg Playtime: <strong>{t_playtime:.1f} hrs</strong> &nbsp;|&nbsp;
            🌐 Languages: <strong>{int(target_profile.get('languages_count', 1))}</strong> &nbsp;|&nbsp;
            💻 Platforms: <strong>{int(target_profile.get('patforms_count', 1))}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Query Similarity Engine ───────────────────────────────────────────────
    exclude_id = int(target_row["app_id"]) if "app_id" in target_row and not pd.isna(target_row["app_id"]) else None

    # Filter dataset if genre constraint is active
    query_df = df[df["primary_genre"] == t_genre] if genre_filter else df

    try:
        results_df = find_similar_games(
            game_profile=target_profile,
            df=query_df,
            n=top_n + 5,
            exclude_app_id=exclude_id,
        )
    except Exception as e:
        st.error(f"Error executing similarity query: {e}")
        return

    # Filter by minimum similarity score
    if not results_df.empty:
        results_df = results_df[results_df["similarity_score"] >= (min_sim_pct / 100.0)].head(top_n)

    if results_df.empty:
        st.warning(f"No similar titles found meeting a {min_sim_pct}% similarity threshold. Try lowering the threshold.")
        return

    # ── Analysis Tabs ─────────────────────────────────────────────────────────
    tab_list, tab_radar, tab_scatter, tab_insights = st.tabs([
        "📋 Competitor Cluster Table",
        "🕸️ Overlay Radar Comparison",
        "🗺️ Competitor Feature Space",
        "💡 Pricing & Reception Deltas",
    ])

    # ── Tab 1: Competitor Cluster Table ──────────────────────────────────────
    with tab_list:
        st.markdown('<div class="section-header">Closest Empirical Competitors</div>', unsafe_allow_html=True)

        disp_df = results_df[[
            "name", "similarity_score", "price", "review_score_pct",
            "peak_ccu", "average_playtime_forever", "primary_genre", "price_tier"
        ]].copy()

        disp_df["similarity_score"] = disp_df["similarity_score"].apply(lambda s: f"{s*100:.1f}%")
        disp_df["price"] = disp_df["price"].apply(lambda p: f"${p:.2f}")
        disp_df["review_score_pct"] = disp_df["review_score_pct"].apply(lambda r: f"{r*100:.1f}%")
        disp_df["peak_ccu"] = disp_df["peak_ccu"].apply(lambda c: f"{int(c):,}")
        disp_df["average_playtime_forever"] = disp_df["average_playtime_forever"].apply(lambda m: f"{int(m):,} m")

        disp_df.columns = [
            "Competitor Title", "Similarity", "Price", "Review %",
            "Peak CCU", "Avg Playtime", "Primary Genre", "Price Tier"
        ]

        st.dataframe(disp_df, use_container_width=True, hide_index=True)

        # Download CSV
        csv_buff = io.StringIO()
        results_df.to_csv(csv_buff, index=False)
        st.download_button(
            label="📥 Download Competitor Cluster CSV",
            data=csv_buff.getvalue(),
            file_name=f"competitors_{t_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            key="sim_download_csv",
        )

    # ── Tab 2: Overlay Radar ──────────────────────────────────────────────────
    with tab_radar:
        st.markdown('<div class="section-header">Target vs Top 3 Competitor Overlay</div>', unsafe_allow_html=True)

        radar_cols = ["price", "review_score_pct", "total_review", "peak_ccu",
                      "average_playtime_forever", "languages_count", "patforms_count"]
        all_medians = df[radar_cols].median()

        fig_radar = _render_comparison_radar(
            target_row=target_profile,
            competitor_rows=results_df.to_dict(orient="records"),
            all_medians=all_medians,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Tab 3: Competitor Feature Space (Scatter) ─────────────────────────────
    with tab_scatter:
        st.markdown('<div class="section-header">Competitor Cluster Distribution (Price vs Review Score)</div>', unsafe_allow_html=True)

        # Background sample for market context
        bg_sample = df.sample(min(800, len(df)), random_state=42)

        fig_scat = px.scatter(
            bg_sample,
            x="price",
            y="review_score_pct",
            opacity=0.15,
            color_discrete_sequence=["#4B5563"],
            template="plotly_dark",
            labels={"price": "Price ($ USD)", "review_score_pct": "Review Score (% Positive)"},
            title="Competitor Positioning Relative to Broader Market",
        )

        # Competitors trace
        fig_scat.add_trace(go.Scatter(
            x=results_df["price"],
            y=results_df["review_score_pct"],
            mode="markers+text",
            marker=dict(size=12, color=ACCENT_COLORS[0], line=dict(color="#FFFFFF", width=1.5)),
            text=results_df["name"],
            textposition="top right",
            textfont=dict(color="#E2E8F0", size=10),
            name="Similar Competitors",
        ))

        # Target trace
        fig_scat.add_trace(go.Scatter(
            x=[t_price],
            y=[t_rev / 100.0],
            mode="markers+text",
            marker=dict(size=18, color="#F59E0B", symbol="star", line=dict(color="#FFFFFF", width=2)),
            text=[f"★ {t_name}"],
            textposition="bottom center",
            textfont=dict(color="#F59E0B", size=13, family="Inter"),
            name="Target Game",
        ))

        max_p = max(results_df["price"].max(), t_price) * 1.2
        fig_scat.update_layout(
            paper_bgcolor=PLOTLY_PAPER_BG,
            plot_bgcolor=PLOTLY_BG_COLOR,
            font=dict(color="#e0e0e0", family="Inter, sans-serif"),
            xaxis=dict(range=[0, max(max_p, 40)]),
            yaxis=dict(range=[0.4, 1.02]),
            margin=dict(t=50, b=40, l=40, r=40),
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    # ── Tab 4: Strategic Deltas ───────────────────────────────────────────────
    with tab_insights:
        st.markdown('<div class="section-header">Commercial & Quality Gap Analysis</div>', unsafe_allow_html=True)

        comp_med_price = float(results_df["price"].median())
        comp_med_rev = float(results_df["review_score_pct"].median() * 100.0)
        comp_med_ccu = float(results_df["peak_ccu"].median())
        comp_med_pt = float(results_df["average_playtime_forever"].median() / 60.0)

        price_diff = round(t_price - comp_med_price, 2)
        rev_diff = round(t_rev - comp_med_rev, 1)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "Price vs Peer Median",
                f"${t_price:.2f}",
                delta=f"{'+' if price_diff >= 0 else ''}{price_diff:.2f} vs peer median (${comp_med_price:.2f})",
                delta_color="inverse",
            )
        with c2:
            st.metric(
                "Review % vs Peer Median",
                f"{t_rev:.1f}%",
                delta=f"{'+' if rev_diff >= 0 else ''}{rev_diff:.1f}% vs peer median ({comp_med_rev:.1f}%)",
            )
        with c3:
            st.metric("Peer Median Peak CCU", f"{int(comp_med_ccu):,}")
        with c4:
            st.metric("Peer Median Playtime", f"{comp_med_pt:.1f} hrs")

        st.markdown("""
        <div class="info-box" style="margin-top:16px;">
            <strong>Competitor Strategy Takeaway:</strong>
            Evaluating peer cluster differences enables publishers and indie developers to spot competitive advantages.
            Titles priced below their similarity cluster with comparable or superior review scores represent prime candidates for positive word-of-mouth conversion.
        </div>
        """, unsafe_allow_html=True)
