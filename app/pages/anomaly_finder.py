"""
anomaly_finder.py -- Statistical & Machine Learning Anomaly Finder.

Identifies:
  1. Hidden Gems: Stellar review sentiment (>=90%), strong playtime, overlooked review base
  2. Viral Breakout Hits: Disproportionate hyper-scale ownership and CCU
  3. Pricing & Value Anomalies: Extreme price-to-quality disconnects (Ultra Bargains vs Overpriced Underperformers)
  4. Engagement Sleepers: Unusually high hours per player relative to catalog visibility
  5. ML Isolation Forest Outliers: Unsupervised multidimensional anomalies in feature space
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest

from src.config import (
    ACCENT_COLORS,
    PRIMARY_GENRES,
    PLOTLY_BG_COLOR,
    PLOTLY_PAPER_BG,
    PRICE_TIER_LABELS,
)


@st.cache_data(show_spinner="Fitting Isolation Forest anomaly detector…", ttl=3600)
def compute_isolation_forest_anomalies(df: pd.DataFrame, contamination: float = 0.015) -> pd.DataFrame:
    """Run unsupervised IsolationForest to detect multidimensional outliers."""
    feature_cols = [
        "price", "review_score_pct", "peak_ccu",
        "average_playtime_forever", "patforms_count", "languages_count"
    ]
    clean = df[feature_cols + ["app_id", "name", "primary_genre", "price_tier", "total_review"]].dropna(subset=feature_cols).copy()

    X = clean[feature_cols].copy()
    X["peak_ccu"] = np.log1p(X["peak_ccu"])
    X["average_playtime_forever"] = np.log1p(X["average_playtime_forever"])

    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    preds = iso.fit_predict(X)
    scores = -iso.score_samples(X)

    # Scale anomaly score to 0–100
    s_min, s_max = scores.min(), scores.max()
    norm_scores = ((scores - s_min) / (s_max - s_min) * 100.0).round(1) if s_max > s_min else scores

    clean["anomaly_score"] = norm_scores
    clean["is_anomaly"] = preds == -1
    return clean[clean["is_anomaly"]].sort_values("anomaly_score", ascending=False).reset_index(drop=True)


def render(df: pd.DataFrame, hide_header: bool = False) -> None:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">Statistical & ML Anomaly Finder</div>
        <div class="hero-subtitle">
            Uncover catalog outliers — discover hidden gems, viral breakout phenomenons,
            pricing-to-quality disconnects, engagement sleepers, and multidimensional Isolation Forest anomalies.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Global Genre & Scope Filter ──────────────────────────────────────────
    with st.expander("🔍 Filter Scope & Thresholds", expanded=True):
        col_g, col_min_rev = st.columns(2)
        with col_g:
            genre_filter = st.selectbox(
                "Filter by Primary Genre:",
                ["All Genres"] + PRIMARY_GENRES,
                index=0,
                key="anomaly_genre_select",
            )
        with col_min_rev:
            min_rev_threshold = st.slider(
                "Minimum User Reviews Filter:",
                min_value=10,
                max_value=1000,
                value=50,
                step=10,
                key="anomaly_min_rev",
            )

    scope_df = df[df["total_review"] >= min_rev_threshold].copy()
    if genre_filter != "All Genres":
        scope_df = scope_df[scope_df["primary_genre"] == genre_filter]

    if scope_df.empty:
        st.warning("No games found matching current filters.")
        return

    # ── Anomaly Tabs ─────────────────────────────────────────────────────────
    tab_gems, tab_viral, tab_pricing, tab_sleepers, tab_ml = st.tabs([
        "💎 Hidden Gems",
        "🚀 Viral Breakout Hits",
        "🏷️ Pricing Disconnects",
        "⏳ Engagement Sleepers",
        "🌲 ML Isolation Forest",
    ])

    # ── Tab 1: Hidden Gems ───────────────────────────────────────────────────
    with tab_gems:
        st.markdown('<div class="section-header">Hidden Gems (High Sentiment · Modest Player Base)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Definition:</strong> Titles with <strong>>= 90% positive reviews</strong>, healthy engagement
            (>= 2 hours playtime), but under <strong>3,000 total reviews</strong>. These titles are critically adored by their audience
            but remain commercially underserved.
        </div>
        """, unsafe_allow_html=True)

        gems_df = scope_df[
            (scope_df["review_score_pct"] >= 0.90) &
            (scope_df["total_review"] <= 3000) &
            (scope_df["average_playtime_forever"] >= 120) &
            (scope_df["price"] > 0)
        ].sort_values(["review_score_pct", "total_review"], ascending=[False, False])

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Identified Hidden Gems", f"{len(gems_df):,}")
        with k2:
            st.metric("Median Gem Price", f"${gems_df['price'].median():.2f}" if not gems_df.empty else "N/A")
        with k3:
            st.metric("Median Review Score", f"{gems_df['review_score_pct'].median()*100:.1f}%" if not gems_df.empty else "N/A")

        if not gems_df.empty:
            fig_gem = px.scatter(
                gems_df.head(200),
                x="price",
                y="review_score_pct",
                size="total_review",
                color="primary_genre",
                hover_name="name",
                template="plotly_dark",
                color_discrete_sequence=ACCENT_COLORS,
                title="Hidden Gems: Price vs Review Score (% Positive, size = Reviews)",
                labels={"price": "Price ($ USD)", "review_score_pct": "Review Score (% Positive)"},
            )
            fig_gem.update_layout(paper_bgcolor=PLOTLY_PAPER_BG, plot_bgcolor=PLOTLY_BG_COLOR)
            st.plotly_chart(fig_gem, use_container_width=True)

            disp_gems = gems_df[[
                "name", "price", "review_score_pct", "total_review",
                "average_playtime_forever", "primary_genre", "price_tier"
            ]].head(50).copy()
            disp_gems["price"] = disp_gems["price"].apply(lambda p: f"${p:.2f}")
            disp_gems["review_score_pct"] = disp_gems["review_score_pct"].apply(lambda r: f"{r*100:.1f}%")
            disp_gems["total_review"] = disp_gems["total_review"].apply(lambda r: f"{int(r):,}")
            disp_gems["average_playtime_forever"] = disp_gems["average_playtime_forever"].apply(lambda m: f"{float(m)/60:.1f}h")
            disp_gems.columns = ["Game Title", "Price", "Positive %", "Reviews", "Avg Playtime", "Genre", "Price Tier"]

            st.dataframe(disp_gems, use_container_width=True, hide_index=True)

            csv_b = io.StringIO()
            gems_df.to_csv(csv_b, index=False)
            st.download_button("📥 Download Hidden Gems CSV", csv_b.getvalue(), "hidden_gems.csv", "text/csv", key="dl_gems")

    # ── Tab 2: Viral Breakouts ───────────────────────────────────────────────
    with tab_viral:
        st.markdown('<div class="section-header">Viral Breakout Phenomenons (Hyper-Scale Outliers)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Definition:</strong> Titles with massive scale (<strong>>= 25,000 user reviews</strong> or <strong>>= 10,000 peak CCU</strong>)
            delivering outsized cultural impact relative to normal catalog releases.
        </div>
        """, unsafe_allow_html=True)

        viral_df = scope_df[
            (scope_df["total_review"] >= 25000) |
            (scope_df["peak_ccu"] >= 10000)
        ].sort_values("total_review", ascending=False)

        vk1, vk2, vk3 = st.columns(3)
        with vk1:
            st.metric("Viral Titles Identified", f"{len(viral_df):,}")
        with vk2:
            st.metric("Median Viral Price", f"${viral_df['price'].median():.2f}" if not viral_df.empty else "N/A")
        with vk3:
            st.metric("Median Peak CCU", f"{int(viral_df['peak_ccu'].median()):,}" if not viral_df.empty else "N/A")

        if not viral_df.empty:
            fig_viral = px.scatter(
                viral_df.head(100),
                x="peak_ccu",
                y="total_review",
                size="owners_mid",
                color="primary_genre",
                hover_name="name",
                log_x=True,
                log_y=True,
                template="plotly_dark",
                color_discrete_sequence=ACCENT_COLORS,
                title="Viral Scale: Peak CCU vs Total Reviews (Log-Log Scale)",
                labels={"peak_ccu": "Peak CCU (Log)", "total_review": "Total Reviews (Log)"},
            )
            fig_viral.update_layout(paper_bgcolor=PLOTLY_PAPER_BG, plot_bgcolor=PLOTLY_BG_COLOR)
            st.plotly_chart(fig_viral, use_container_width=True)

            disp_viral = viral_df[[
                "name", "price", "review_score_pct", "total_review",
                "peak_ccu", "owners_mid", "primary_genre"
            ]].head(40).copy()
            disp_viral["price"] = disp_viral["price"].apply(lambda p: f"${p:.2f}")
            disp_viral["review_score_pct"] = disp_viral["review_score_pct"].apply(lambda r: f"{r*100:.1f}%")
            disp_viral["total_review"] = disp_viral["total_review"].apply(lambda r: f"{int(r):,}")
            disp_viral["peak_ccu"] = disp_viral["peak_ccu"].apply(lambda c: f"{int(c):,}")
            disp_viral["owners_mid"] = disp_viral["owners_mid"].apply(lambda o: f"{int(o):,}")
            disp_viral.columns = ["Game Title", "Price", "Positive %", "Reviews", "Peak CCU", "Est Owners", "Genre"]

            st.dataframe(disp_viral, use_container_width=True, hide_index=True)

            csv_v = io.StringIO()
            viral_df.to_csv(csv_v, index=False)
            st.download_button("📥 Download Viral Outliers CSV", csv_v.getvalue(), "viral_breakouts.csv", "text/csv", key="dl_viral")

    # ── Tab 3: Pricing Disconnects ───────────────────────────────────────────
    with tab_pricing:
        st.markdown('<div class="section-header">Pricing-to-Quality Disconnects</div>', unsafe_allow_html=True)
        disconnect_mode = st.radio(
            "Select Disconnect Type:",
            ["Super Bargains (High Quality, Very Low Price)", "Overpriced / Underdelivered (High Price, Poor Sentiment)"],
            horizontal=True,
            key="anomaly_pricing_mode",
        )

        if "Super Bargains" in disconnect_mode:
            bargain_df = scope_df[
                (scope_df["price"] > 0) &
                (scope_df["price"] <= 4.99) &
                (scope_df["review_score_pct"] >= 0.90) &
                (scope_df["total_review"] >= 200)
            ].sort_values("review_score_pct", ascending=False)

            st.markdown(f"Found **{len(bargain_df):,}** ultra-high-value titles priced at or below $4.99 with >= 90% positive sentiment.")
            disp_b = bargain_df[["name", "price", "review_score_pct", "total_review", "average_playtime_forever", "primary_genre"]].head(40).copy()
            disp_b["price"] = disp_b["price"].apply(lambda p: f"${p:.2f}")
            disp_b["review_score_pct"] = disp_b["review_score_pct"].apply(lambda r: f"{r*100:.1f}%")
            disp_b["total_review"] = disp_b["total_review"].apply(lambda r: f"{int(r):,}")
            disp_b["average_playtime_forever"] = disp_b["average_playtime_forever"].apply(lambda m: f"{float(m)/60:.1f}h")
            disp_b.columns = ["Game Title", "Price", "Positive %", "Reviews", "Avg Playtime", "Genre"]
            st.dataframe(disp_b, use_container_width=True, hide_index=True)

        else:
            overpriced_df = scope_df[
                (scope_df["price"] >= 29.99) &
                (scope_df["review_score_pct"] <= 0.55)
            ].sort_values("review_score_pct", ascending=True)

            st.markdown(f"Found **{len(overpriced_df):,}** premium-priced titles (>= $29.99) suffering low review sentiment (<= 55%).")
            disp_o = overpriced_df[["name", "price", "review_score_pct", "total_review", "primary_genre", "price_tier"]].head(40).copy()
            disp_o["price"] = disp_o["price"].apply(lambda p: f"${p:.2f}")
            disp_o["review_score_pct"] = disp_o["review_score_pct"].apply(lambda r: f"{r*100:.1f}%")
            disp_o["total_review"] = disp_o["total_review"].apply(lambda r: f"{int(r):,}")
            disp_o.columns = ["Game Title", "Price", "Positive %", "Reviews", "Genre", "Tier"]
            st.dataframe(disp_o, use_container_width=True, hide_index=True)

    # ── Tab 4: Engagement Sleepers ───────────────────────────────────────────
    with tab_sleepers:
        st.markdown('<div class="section-header">Engagement Sleepers (Astronomical Playtime per Owner)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Definition:</strong> Titles where average playtime exceeds <strong>50 hours</strong> (3,000+ minutes).
            These represent deeply sticky titles with exceptional player retention (simulation, grand strategy, deep survival).
        </div>
        """, unsafe_allow_html=True)

        sleepers_df = scope_df[
            (scope_df["average_playtime_forever"] >= 3000)
        ].sort_values("average_playtime_forever", ascending=False)

        st.metric("Engagement Sleepers (> 50 hrs avg)", f"{len(sleepers_df):,}")

        if not sleepers_df.empty:
            disp_s = sleepers_df[["name", "average_playtime_forever", "price", "review_score_pct", "total_review", "primary_genre"]].head(40).copy()
            disp_s["average_playtime_forever"] = disp_s["average_playtime_forever"].apply(lambda m: f"{float(m)/60:.1f} hrs")
            disp_s["price"] = disp_s["price"].apply(lambda p: f"${p:.2f}")
            disp_s["review_score_pct"] = disp_s["review_score_pct"].apply(lambda r: f"{r*100:.1f}%")
            disp_s["total_review"] = disp_s["total_review"].apply(lambda r: f"{int(r):,}")
            disp_s.columns = ["Game Title", "Avg Playtime", "Price", "Positive %", "Reviews", "Genre"]
            st.dataframe(disp_s, use_container_width=True, hide_index=True)

            csv_s = io.StringIO()
            sleepers_df.to_csv(csv_s, index=False)
            st.download_button("📥 Download Engagement Sleepers CSV", csv_s.getvalue(), "engagement_sleepers.csv", "text/csv", key="dl_sleepers")

    # ── Tab 5: ML Isolation Forest Outliers ───────────────────────────────────
    with tab_ml:
        st.markdown('<div class="section-header">Multidimensional Isolation Forest Outliers</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Methodology:</strong> Scikit-learn's <em>IsolationForest</em> algorithm isolates observations
            by randomly selecting a feature and splitting value. Outliers require significantly fewer splits to isolate in multidimensional space.
        </div>
        """, unsafe_allow_html=True)

        contam = st.slider("Anomaly Contamination Rate (%):", 0.5, 5.0, 1.5, step=0.5, key="anomaly_contam_rate") / 100.0

        try:
            iso_anomalies = compute_isolation_forest_anomalies(scope_df, contamination=contam)
            st.metric("Machine Learning Anomalies Flagged", f"{len(iso_anomalies):,}")

            if not iso_anomalies.empty:
                fig_iso = px.histogram(
                    iso_anomalies,
                    x="anomaly_score",
                    nbins=25,
                    color_discrete_sequence=[ACCENT_COLORS[0]],
                    template="plotly_dark",
                    title="Isolation Forest Anomaly Score Distribution (Higher = More Anomalous)",
                    labels={"anomaly_score": "Anomaly Score (0 - 100)"},
                )
                fig_iso.update_layout(paper_bgcolor=PLOTLY_PAPER_BG, plot_bgcolor=PLOTLY_BG_COLOR)
                st.plotly_chart(fig_iso, use_container_width=True)

                disp_iso = iso_anomalies[[
                    "name", "anomaly_score", "price", "review_score_pct",
                    "total_review", "peak_ccu", "primary_genre"
                ]].head(50).copy()
                disp_iso["anomaly_score"] = disp_iso["anomaly_score"].apply(lambda s: f"{s:.1f} / 100")
                disp_iso["price"] = disp_iso["price"].apply(lambda p: f"${p:.2f}")
                disp_iso["review_score_pct"] = disp_iso["review_score_pct"].apply(lambda r: f"{r*100:.1f}%")
                disp_iso["total_review"] = disp_iso["total_review"].apply(lambda r: f"{int(r):,}")
                disp_iso["peak_ccu"] = disp_iso["peak_ccu"].apply(lambda c: f"{int(c):,}")
                disp_iso.columns = ["Game Title", "Anomaly Score", "Price", "Positive %", "Reviews", "Peak CCU", "Genre"]

                st.dataframe(disp_iso, use_container_width=True, hide_index=True)

                csv_iso = io.StringIO()
                iso_anomalies.to_csv(csv_iso, index=False)
                st.download_button("📥 Download ML Anomalies CSV", csv_iso.getvalue(), "isolation_forest_anomalies.csv", "text/csv", key="dl_iso")
        except Exception as e:
            st.error(f"Error computing Isolation Forest anomalies: {e}")
