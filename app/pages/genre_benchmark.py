"""
genre_benchmark.py -- Deep Genre Benchmark & Competition Density Explorer.

Features:
  - Genre-level competitive intelligence & KPI percentiles
  - Pricing & monetization distributions (Price Tier breakdown, price boxplots)
  - Player engagement & quality distributions
  - Competition Density Heatmap (Genre x Price Tier)
  - Empirical Market Gap Signals (demand vs supply indicators)
  - Annual release density trends
  - Top benchmark games per genre with CSV download
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import genre_summary
from src.benchmarks import compute_genre_benchmarks
from src.config import (
    ACCENT_COLORS,
    PRIMARY_GENRES,
    PLOTLY_BG_COLOR,
    PLOTLY_PAPER_BG,
    PRICE_TIER_LABELS,
)
from src.market_position import (
    competition_genre_x_tier,
    compute_market_gap_signal,
    release_density_by_genre_year,
)


def render(df: pd.DataFrame) -> None:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">Genre Benchmark & Competition Density</div>
        <div class="hero-subtitle">
            Deep genre intelligence — analyze competitive saturation, pricing distributions,
            player engagement benchmarks, and empirical market gap signals across Steam genres.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Controls ─────────────────────────────────────────────────────────────
    with st.expander("🔍 Filter & Comparison Settings", expanded=True):
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            selected_genre = st.selectbox(
                "Primary Genre to Benchmark:",
                PRIMARY_GENRES,
                index=0,
                key="genre_bench_primary",
            )

        with col_c2:
            compare_mode = st.checkbox("Compare with a second genre", value=False, key="genre_bench_comp_toggle")
            if compare_mode:
                sec_default = 1 if len(PRIMARY_GENRES) > 1 else 0
                compare_genre = st.selectbox(
                    "Comparison Genre:",
                    [g for g in PRIMARY_GENRES if g != selected_genre],
                    index=0,
                    key="genre_bench_secondary",
                )
            else:
                compare_genre = None

        with col_c3:
            min_reviews = st.selectbox(
                "Minimum Review Threshold:",
                [0, 10, 50, 100, 500],
                index=1,
                format_func=lambda x: "All Games (0+ reviews)" if x == 0 else f"Commercial / Established ({x}+ reviews)",
                key="genre_bench_min_rev",
            )

    # Filter dataset for analysis
    filtered_df = df[df["total_review"] >= min_reviews] if min_reviews > 0 else df
    genre_data = filtered_df[filtered_df["primary_genre"] == selected_genre]
    total_market_games = len(filtered_df)
    genre_games_count = len(genre_data)

    if genre_games_count == 0:
        st.warning(f"No games found in {selected_genre} with the selected review threshold.")
        return

    # Calculate Market Deltas
    market_share_pct = (genre_games_count / total_market_games * 100.0) if total_market_games > 0 else 0.0
    genre_med_price = float(genre_data["price"].median())
    market_med_price = float(filtered_df["price"].median())
    price_delta = round(genre_med_price - market_med_price, 2)

    genre_med_quality = float(genre_data["review_score_pct"].dropna().median() * 100.0)
    market_med_quality = float(filtered_df["review_score_pct"].dropna().median() * 100.0)
    quality_delta = round(genre_med_quality - market_med_quality, 1)

    genre_med_ccu = float(genre_data["peak_ccu"].median())
    genre_med_playtime = float(genre_data["average_playtime_forever"].median())

    # ── KPI Cards ────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{genre_games_count:,}</div>
            <div class="metric-label">{selected_genre} Titles ({market_share_pct:.1f}% Market)</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${genre_med_price:.2f}</div>
            <div class="metric-label">Median Price ({'+' if price_delta >= 0 else ''}{price_delta:.2f} vs Market)</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{genre_med_quality:.1f}%</div>
            <div class="metric-label">Median Review Score ({'+' if quality_delta >= 0 else ''}{quality_delta:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{int(genre_med_ccu):,}</div>
            <div class="metric-label">Median Peak CCU</div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{genre_med_playtime:.0f}m</div>
            <div class="metric-label">Median Playtime ({genre_med_playtime/60:.1f}h)</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_landscape, tab_pricing, tab_engagement, tab_density, tab_top_games = st.tabs([
        "📊 Competitive Landscape",
        "💰 Pricing & Monetization",
        "📈 Quality & Engagement",
        "🗺️ Competition Heatmap & Gaps",
        "🏆 Top Benchmark Titles",
    ])

    # ── Tab 1: Competitive Landscape ─────────────────────────────────────────
    with tab_landscape:
        st.markdown('<div class="section-header">Steam Market Genre Distribution</div>', unsafe_allow_html=True)

        g_sum = genre_summary(filtered_df, min_games=10)
        g_sum["is_selected"] = g_sum["primary_genre"] == selected_genre
        if compare_mode and compare_genre:
            g_sum["is_compare"] = g_sum["primary_genre"] == compare_genre
        else:
            g_sum["is_compare"] = False

        col_g1, col_g2 = st.columns([1.2, 0.8])
        with col_g1:
            # Color map for bar chart
            colors = []
            for _, r in g_sum.iterrows():
                if r["is_selected"]:
                    colors.append(ACCENT_COLORS[0])
                elif r["is_compare"]:
                    colors.append(ACCENT_COLORS[1])
                else:
                    colors.append("#2D3748")

            fig_bar = go.Figure(go.Bar(
                x=g_sum["game_count"],
                y=g_sum["primary_genre"],
                orientation="h",
                marker_color=colors,
                text=g_sum["game_count"].apply(lambda x: f"{x:,}"),
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="Active Titles by Primary Genre (Current Selection Highlighted)",
                template="plotly_dark",
                paper_bgcolor=PLOTLY_PAPER_BG,
                plot_bgcolor=PLOTLY_BG_COLOR,
                font=dict(color="#e0e0e0", family="Inter, sans-serif"),
                xaxis=dict(title="Game Count"),
                yaxis=dict(title="", autorange="reversed"),
                margin=dict(t=50, b=40, l=100, r=40),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_g2:
            st.markdown(f"**Key Metrics Comparison: {selected_genre}**")
            summary_disp = g_sum[[
                "primary_genre", "game_count", "median_price",
                "median_quality", "mean_playtime_hrs", "mean_platforms"
            ]].copy()
            summary_disp.columns = ["Genre", "Games", "Med Price", "Med Review %", "Avg Play (h)", "Avg Platforms"]
            summary_disp["Med Price"] = summary_disp["Med Price"].apply(lambda p: f"${p:.2f}")
            summary_disp["Med Review %"] = summary_disp["Med Review %"].apply(lambda q: f"{q:.1f}%")
            summary_disp["Avg Play (h)"] = summary_disp["Avg Play (h)"].apply(lambda h: f"{h:.1f}h")
            summary_disp["Avg Platforms"] = summary_disp["Avg Platforms"].apply(lambda pl: f"{pl:.1f}")

            st.dataframe(summary_disp, use_container_width=True, hide_index=True)

    # ── Tab 2: Pricing & Monetization ────────────────────────────────────────
    with tab_pricing:
        st.markdown(f'<div class="section-header">Pricing Structure in {selected_genre}</div>', unsafe_allow_html=True)
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            # Price Tier composition
            tier_order = ["Free", "Budget", "Mid-range", "Premium", "AAA"]
            genre_tiers = genre_data["price_tier"].value_counts().reindex(tier_order, fill_value=0).reset_index()
            genre_tiers.columns = ["Tier", "Count"]
            genre_tiers["Pct"] = (genre_tiers["Count"] / genre_tiers["Count"].sum() * 100).round(1)

            fig_tier = px.bar(
                genre_tiers,
                x="Tier",
                y="Count",
                text=genre_tiers["Pct"].apply(lambda x: f"{x:.1f}%"),
                template="plotly_dark",
                color="Tier",
                color_discrete_sequence=ACCENT_COLORS,
                title=f"Price Tier Breakdown in {selected_genre}",
                labels={"Count": "Game Count", "Tier": "Price Tier"},
            )
            fig_tier.update_traces(textposition="outside")
            fig_tier.update_layout(
                paper_bgcolor=PLOTLY_PAPER_BG,
                plot_bgcolor=PLOTLY_BG_COLOR,
                font=dict(color="#e0e0e0", family="Inter, sans-serif"),
                showlegend=False,
                margin=dict(t=50, b=40, l=40, r=40),
            )
            st.plotly_chart(fig_tier, use_container_width=True)

        with col_p2:
            # Price Distribution Boxplot / Comparison
            comp_dfs = [genre_data.assign(Scope=selected_genre)]
            if compare_mode and compare_genre:
                comp_dfs.append(filtered_df[filtered_df["primary_genre"] == compare_genre].assign(Scope=compare_genre))
            comp_dfs.append(filtered_df.assign(Scope="All Steam"))
            box_data = pd.concat(comp_dfs, ignore_index=True)
            box_data = box_data[box_data["price"] <= 70.0]

            fig_box = px.box(
                box_data,
                x="Scope",
                y="price",
                color="Scope",
                template="plotly_dark",
                color_discrete_sequence=[ACCENT_COLORS[0], ACCENT_COLORS[1], "#6B7280"],
                title="Launch Price Distribution (titles <= $70)",
                labels={"price": "Price ($ USD)", "Scope": "Cohort"},
            )
            fig_box.update_layout(
                paper_bgcolor=PLOTLY_PAPER_BG,
                plot_bgcolor=PLOTLY_BG_COLOR,
                font=dict(color="#e0e0e0", family="Inter, sans-serif"),
                showlegend=False,
                margin=dict(t=50, b=40, l=40, r=40),
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # Revenue proxy / Estimated Owners across tiers
        st.markdown(f'<div class="section-header">Estimated Ownership by Price Tier in {selected_genre}</div>', unsafe_allow_html=True)
        tier_stats = genre_data.groupby("price_tier").agg(
            games=("app_id", "count"),
            median_owners=("owners_mid", "median"),
            mean_owners=("owners_mid", "mean"),
            median_ccu=("peak_ccu", "median"),
            median_reviews=("total_review", "median"),
        ).reindex(tier_order).dropna(subset=["games"]).reset_index()

        tier_disp = tier_stats.copy()
        tier_disp.columns = ["Price Tier", "Games", "Median Owners", "Mean Owners", "Median CCU", "Median Reviews"]
        tier_disp["Median Owners"] = tier_disp["Median Owners"].apply(lambda o: f"{int(o):,}")
        tier_disp["Mean Owners"] = tier_disp["Mean Owners"].apply(lambda o: f"{int(o):,}")
        tier_disp["Median CCU"] = tier_disp["Median CCU"].apply(lambda c: f"{int(c):,}")
        tier_disp["Median Reviews"] = tier_disp["Median Reviews"].apply(lambda r: f"{int(r):,}")
        st.dataframe(tier_disp, use_container_width=True, hide_index=True)

    # ── Tab 3: Quality & Engagement ──────────────────────────────────────────
    with tab_engagement:
        st.markdown(f'<div class="section-header">Review Score & Player Engagement</div>', unsafe_allow_html=True)
        col_e1, col_e2 = st.columns(2)

        with col_e1:
            # Review distribution
            rev_scores = (genre_data["review_score_pct"].dropna() * 100.0)
            fig_rev = px.histogram(
                rev_scores,
                nbins=30,
                template="plotly_dark",
                color_discrete_sequence=[ACCENT_COLORS[0]],
                title=f"Review Score Distribution in {selected_genre}",
                labels={"value": "Positive Review Score (%)", "count": "Game Count"},
            )
            fig_rev.add_vline(
                x=genre_med_quality,
                line_dash="dash",
                line_color="#FFFFFF",
                annotation_text=f"Median ({genre_med_quality:.1f}%)",
                annotation_position="top left",
            )
            fig_rev.update_layout(
                paper_bgcolor=PLOTLY_PAPER_BG,
                plot_bgcolor=PLOTLY_BG_COLOR,
                font=dict(color="#e0e0e0", family="Inter, sans-serif"),
                margin=dict(t=50, b=40, l=40, r=40),
            )
            st.plotly_chart(fig_rev, use_container_width=True)

        with col_e2:
            # Playtime distribution
            pt_data = genre_data[genre_data["average_playtime_forever"] > 0]["average_playtime_forever"] / 60.0
            fig_pt = px.histogram(
                pt_data.clip(upper=50.0),
                nbins=25,
                template="plotly_dark",
                color_discrete_sequence=[ACCENT_COLORS[1]],
                title=f"Player Playtime Distribution in {selected_genre} (Capped at 50h)",
                labels={"value": "Average Playtime (Hours)", "count": "Game Count"},
            )
            fig_pt.update_layout(
                paper_bgcolor=PLOTLY_PAPER_BG,
                plot_bgcolor=PLOTLY_BG_COLOR,
                font=dict(color="#e0e0e0", family="Inter, sans-serif"),
                margin=dict(t=50, b=40, l=40, r=40),
            )
            st.plotly_chart(fig_pt, use_container_width=True)

        # Scatter plot: Price vs Quality in this genre
        st.markdown(f'<div class="section-header">Price vs Review Score Matrix ({selected_genre})</div>', unsafe_allow_html=True)
        scatter_sample = genre_data.sample(min(len(genre_data), 1200), random_state=42)
        fig_pq = px.scatter(
            scatter_sample,
            x="price",
            y="review_score_pct",
            color="price_tier",
            size="peak_ccu",
            hover_name="name",
            template="plotly_dark",
            color_discrete_sequence=ACCENT_COLORS,
            title="Price vs Review Score (% Positive) with CCU Bubble Size",
            labels={"price": "Price ($ USD)", "review_score_pct": "Review Score (% Positive)", "price_tier": "Tier"},
        )
        fig_pq.add_vline(x=genre_med_price, line_dash="dot", line_color="#9CA3AF")
        fig_pq.add_hline(y=genre_med_quality / 100.0, line_dash="dot", line_color="#9CA3AF")
        fig_pq.update_layout(
            paper_bgcolor=PLOTLY_PAPER_BG,
            plot_bgcolor=PLOTLY_BG_COLOR,
            font=dict(color="#e0e0e0", family="Inter, sans-serif"),
            margin=dict(t=50, b=40, l=40, r=40),
        )
        st.plotly_chart(fig_pq, use_container_width=True)

    # ── Tab 4: Competition Heatmap & Gaps ────────────────────────────────────
    with tab_density:
        st.markdown('<div class="section-header">Competition Density Heatmap (Genre × Price Tier)</div>', unsafe_allow_html=True)
        st.markdown("Crowding analysis across Steam segments — shows where titles congregate:")

        density_pivot = competition_genre_x_tier(filtered_df, min_genre_games=30)
        fig_heat = px.imshow(
            density_pivot,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Purples",
            template="plotly_dark",
            title="Game Count by Genre and Price Tier",
            labels=dict(x="Price Tier", y="Primary Genre", color="Titles"),
        )
        fig_heat.update_layout(
            paper_bgcolor=PLOTLY_PAPER_BG,
            plot_bgcolor=PLOTLY_BG_COLOR,
            font=dict(color="#e0e0e0", family="Inter, sans-serif"),
            margin=dict(t=50, b=40, l=120, r=40),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        col_gap, col_trend = st.columns(2)

        with col_gap:
            st.markdown('<div class="section-header">Market Gap Signals (Observed Demand vs Supply)</div>', unsafe_allow_html=True)
            gap_signals = compute_market_gap_signal(filtered_df, min_games=30)
            if not gap_signals.empty:
                gap_top = gap_signals.head(10)[["primary_genre", "price_tier", "game_count", "median_owners", "gap_signal_norm"]]
                gap_top.columns = ["Genre", "Tier", "Supply (Games)", "Median Owners", "Gap Signal (0-100)"]
                gap_top["Median Owners"] = gap_top["Median Owners"].apply(lambda o: f"{int(o):,}")
                st.dataframe(gap_top, use_container_width=True, hide_index=True)
                st.markdown("""
                <div class="info-box" style="font-size:0.8rem;">
                    <strong>Gap Signal Methodology:</strong> Evaluates observed demand (Median Owners × Review Score)
                    divided by supply density (Compromising game count). High scores reflect segments with observed appetite but lower competitive crowding.
                </div>
                """, unsafe_allow_html=True)

        with col_trend:
            st.markdown(f'<div class="section-header">Annual Release Density ({selected_genre})</div>', unsafe_allow_html=True)
            genre_releases = release_density_by_genre_year(filtered_df, top_genres=10)
            if selected_genre in genre_releases.columns:
                rel_series = genre_releases[selected_genre].reset_index()
                rel_series.columns = ["Year", "Releases"]
                rel_series = rel_series[rel_series["Year"] >= 2010]

                fig_trend = px.bar(
                    rel_series,
                    x="Year",
                    y="Releases",
                    template="plotly_dark",
                    color_discrete_sequence=[ACCENT_COLORS[0]],
                    title=f"Annual Releases in {selected_genre} (2010 - Present)",
                )
                fig_trend.update_layout(
                    paper_bgcolor=PLOTLY_PAPER_BG,
                    plot_bgcolor=PLOTLY_BG_COLOR,
                    font=dict(color="#e0e0e0", family="Inter, sans-serif"),
                    margin=dict(t=50, b=40, l=40, r=40),
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info(f"Historical release density data for {selected_genre} is insufficient.")

    # ── Tab 5: Top Benchmark Titles ──────────────────────────────────────────
    with tab_top_games:
        st.markdown(f'<div class="section-header">Top Performing Benchmark Titles in {selected_genre}</div>', unsafe_allow_html=True)

        col_sort, col_top_n = st.columns([2, 1])
        with col_sort:
            sort_metric = st.selectbox(
                "Rank By Metric:",
                ["owners_mid", "peak_ccu", "total_review", "review_score_pct", "price"],
                format_func=lambda x: {
                    "owners_mid": "Estimated Owners",
                    "peak_ccu": "Peak Concurrent Players (CCU)",
                    "total_review": "Total Reviews",
                    "review_score_pct": "Review Score (% Positive)",
                    "price": "Price ($ USD)",
                }.get(x, x),
                key="genre_bench_top_sort",
            )
        with col_top_n:
            top_n = st.slider("Number of Titles to Display:", 5, 50, 15, key="genre_bench_top_n")

        top_games_df = genre_data.nlargest(top_n, sort_metric)[[
            "name", "price", "review_score_pct", "owners_mid",
            "peak_ccu", "average_playtime_forever", "price_tier"
        ]].copy()

        top_games_disp = top_games_df.copy()
        top_games_disp.columns = ["Title", "Price", "Review Score %", "Est Owners", "Peak CCU", "Avg Playtime (m)", "Price Tier"]
        top_games_disp["Price"] = top_games_disp["Price"].apply(lambda p: f"${p:.2f}")
        top_games_disp["Review Score %"] = top_games_disp["Review Score %"].apply(lambda r: f"{r*100:.1f}%")
        top_games_disp["Est Owners"] = top_games_disp["Est Owners"].apply(lambda o: f"{int(o):,}")
        top_games_disp["Peak CCU"] = top_games_disp["Peak CCU"].apply(lambda c: f"{int(c):,}")
        top_games_disp["Avg Playtime (m)"] = top_games_disp["Avg Playtime (m)"].apply(lambda m: f"{int(m):,}")

        st.dataframe(top_games_disp, use_container_width=True, hide_index=True)

        # Download CSV
        csv_buff = io.StringIO()
        top_games_df.to_csv(csv_buff, index=False)
        st.download_button(
            label="📥 Download Top Genre Benchmarks CSV",
            data=csv_buff.getvalue(),
            file_name=f"genre_benchmarks_{selected_genre.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            key="genre_bench_download",
        )
