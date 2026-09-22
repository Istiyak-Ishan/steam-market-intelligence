"""
segmentation_page.py — K-Means Market Segmentation page.

Segments the Steam market into data-driven clusters using K-Means on
price, quality, ownership, engagement, and platform features.
No hardcoded cluster names — all labels derived from actual cluster profiles.
"""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.segmentation import run_segmentation, get_cluster_profiles
from src.config import SEGMENTATION_FEATURES, ACCENT_COLORS, PLOTLY_BG_COLOR, PLOTLY_PAPER_BG


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pca_scatter(seg_df: pd.DataFrame) -> go.Figure:
    """Compute and plot 2D PCA for visualising cluster separation."""
    feat_cols = [c for c in SEGMENTATION_FEATURES if c in seg_df.columns]
    plot_df = seg_df.dropna(subset=feat_cols + ["cluster_label", "name"]).copy()
    
    if plot_df.empty:
        return go.Figure()
        
    if len(plot_df) > 5000:
        plot_df = plot_df.sample(5000, random_state=42)
        
    X = plot_df[feat_cols]
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    
    plot_df["PC1"] = pcs[:, 0]
    plot_df["PC2"] = pcs[:, 1]
    
    fig = px.scatter(
        plot_df, x="PC1", y="PC2",
        color="cluster_label", opacity=0.6,
        hover_data=["name"], template="plotly_dark",
        color_discrete_sequence=ACCENT_COLORS,
        labels={
            "PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
            "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
            "cluster_label": "Segment"
        }
    )
    fig.update_layout(
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_BG_COLOR,
        font=dict(color="#e0e0e0", family="Inter, sans-serif"),
        margin=dict(t=40, b=40, l=40, r=40),
        title="2D PCA Visualisation of Cluster Boundaries"
    )
    return fig


def _make_heatmap(profiles: pd.DataFrame) -> go.Figure:
    """Generate a normalised heatmap of cluster feature density."""
    feat_cols = [c for c in SEGMENTATION_FEATURES if c in profiles.columns]
    if not feat_cols:
        return go.Figure()

    normed = profiles[feat_cols].copy()
    for col in feat_cols:
        c_min, c_max = normed[col].min(), normed[col].max()
        if c_max > c_min:
            normed[col] = (normed[col] - c_min) / (c_max - c_min)
        else:
            normed[col] = 0.0

    labels = [c.replace("_", " ").title() for c in feat_cols]
    y_labels = profiles["cluster_label"].tolist()

    fig = px.imshow(
        normed.values,
        x=labels,
        y=y_labels,
        color_continuous_scale="Purp",
        aspect="auto",
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_BG_COLOR,
        font=dict(color="#e0e0e0", family="Inter, sans-serif"),
        margin=dict(t=40, b=40, l=40, r=40),
        title="Feature Intensity Heatmap (Normalised 0-1 per feature)"
    )
    return fig


def _make_radar(profiles: pd.DataFrame) -> go.Figure:
    """Radar/spider chart of normalised cluster profiles."""
    feat_cols = [c for c in SEGMENTATION_FEATURES if c in profiles.columns]
    if not feat_cols:
        return go.Figure()

    # Normalise 0-1 per feature for comparability
    normed = profiles[feat_cols].copy()
    for col in feat_cols:
        col_min, col_max = normed[col].min(), normed[col].max()
        rng = col_max - col_min
        normed[col] = (normed[col] - col_min) / rng if rng > 0 else 0.0

    labels = [c.replace("_", " ").title() for c in feat_cols]

    fig = go.Figure()
    for i, row in profiles.iterrows():
        values = normed.loc[i, feat_cols].tolist()
        values_closed = values + [values[0]]
        labels_closed = labels + [labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            name=row["cluster_label"],
            line_color=ACCENT_COLORS[i % len(ACCENT_COLORS)],
            opacity=0.4,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_dark",
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_BG_COLOR,
        title="Cluster Feature Profiles (normalised)",
        font=dict(color="#e0e0e0", family="Inter, sans-serif"),
        margin=dict(t=80, b=40),
    )
    return fig


def _make_scatter(seg_df: pd.DataFrame, x: str, y: str, color_col: str = "cluster_label") -> go.Figure:
    """Scatter plot coloured by cluster."""
    plot_df = seg_df[[x, y, color_col, "name"]].dropna()
    fig = px.scatter(
        plot_df.sample(min(5000, len(plot_df)), random_state=42),
        x=x, y=y,
        color=color_col,
        opacity=0.5,
        hover_data=["name"],
        template="plotly_dark",
        color_discrete_sequence=ACCENT_COLORS,
        labels={
            x: x.replace("_", " ").title(),
            y: y.replace("_", " ").title(),
            color_col: "Segment",
        },
    )
    fig.update_layout(
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_BG_COLOR,
        font=dict(color="#e0e0e0", family="Inter, sans-serif"),
    )
    return fig


# ── Main render ───────────────────────────────────────────────────────────────

def render(df: pd.DataFrame) -> None:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🧩 Market Segmentation</div>
        <div class="hero-subtitle">
            K-Means clustering segments the Steam market into data-driven groups
            based on price, quality, ownership, engagement, and platform reach.
            Cluster labels are derived from actual data — no invented names.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Segmentation Controls**")
        k_choice = st.slider("Number of clusters (k)", min_value=3, max_value=8, value=5)
        force_rerun = st.button("🔄 Force Re-run Segmentation")

    # ── Run / load segmentation ───────────────────────────────────────────────
    with st.spinner("Loading segments…"):
        seg_df = run_segmentation(df, k=k_choice if force_rerun else None, force=force_rerun)

    if "cluster_id" not in seg_df.columns:
        st.error("Segmentation failed — cluster_id column missing. Check src/segmentation.py.")
        return

    profiles = get_cluster_profiles(seg_df)
    n_clusters = seg_df["cluster_id"].nunique()

    # ── Market Archetype Summary ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Market Archetypes Summary</div>', unsafe_allow_html=True)
    summary_md = "Based strictly on the median feature profiles of the generated clusters:<br><ul>"
    for _, row in profiles.iterrows():
        lbl = row['cluster_label']
        price_val = row.get('price', 0)
        rev_val = row.get('review_score_pct', 0) * 100
        play_val = row.get('average_playtime_forever', 0)
        
        summary_md += f"<li><strong>{lbl}</strong> is characterised by a median price of ${price_val:.2f}, {rev_val:.1f}% positive reviews, and {play_val:.0f} mins avg playtime.</li>"
    summary_md += "</ul>"
    
    st.markdown(f'<div class="info-box">{summary_md}</div>', unsafe_allow_html=True)


    # ── KPI row ───────────────────────────────────────────────────────────────
    cols = st.columns(n_clusters)
    for i, (_, row) in enumerate(profiles.iterrows()):
        label = row["cluster_label"]
        count = (seg_df["cluster_id"] == row["cluster_id"]).sum()
        pct   = count / len(seg_df) * 100
        with cols[i % n_clusters]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.4rem;">{label}</div>
                <div class="metric-label">{count:,} games · {pct:.1f}% of market</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_map, tab_profiles, tab_radar, tab_lookup = st.tabs([
        "Cluster Maps", "Cluster Profiles", "Radar Chart", "Game Lookup"
    ])

    # ── TAB 1: Cluster maps ───────────────────────────────────────────────────
    with tab_map:
        st.markdown('<div class="section-header">PCA Cluster Map (2D Projection)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>⚠️ Disclaimer:</strong> PCA is for 2D visualization only; the K-Means clustering was performed 
            on the full high-dimensional scaled feature space. This projection reduces dimensions to show relative boundaries.
        </div>
        """, unsafe_allow_html=True)
        
        fig_pca = _make_pca_scatter(seg_df)
        st.plotly_chart(fig_pca, use_container_width=True)
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-header">Price vs Review Quality</div>', unsafe_allow_html=True)
            fig1 = _make_scatter(seg_df, "price", "review_score_pct")
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown('<div class="section-header">Price vs Estimated Owners</div>', unsafe_allow_html=True)
            fig2 = _make_scatter(seg_df, "price", "owners_mid")
            st.plotly_chart(fig2, use_container_width=True)
            
        with col_b:
            st.markdown('<div class="section-header">Review Quality vs Est. Owners</div>', unsafe_allow_html=True)
            fig3 = _make_scatter(seg_df, "review_score_pct", "owners_mid")
            st.plotly_chart(fig3, use_container_width=True)

            # Average playtime vs price
            st.markdown('<div class="section-header">Avg Playtime vs Price</div>', unsafe_allow_html=True)
            fig4 = _make_scatter(seg_df, "price", "average_playtime_forever")
            st.plotly_chart(fig4, use_container_width=True)

    # ── TAB 2: Cluster profiles table ─────────────────────────────────────────
    with tab_profiles:
        st.markdown('<div class="section-header">Median Feature Values per Cluster</div>', unsafe_allow_html=True)

        display_cols = {
            "cluster_label":            "Segment",
            "price":                    "Median Price ($)",
            "review_score_pct":         "Review Score (ratio)",
            "owners_mid":               "Est. Owners (median)",
            "recommendations":          "Recommendations",
            "peak_ccu":                 "Peak CCU",
            "average_playtime_forever": "Avg Playtime (min)",
            "age_by_years":             "Age (years)",
            "patforms_count":           "Platforms",
            "languages_count":          "Languages",
        }
        avail = {k: v for k, v in display_cols.items() if k in profiles.columns}
        prof_display = profiles[list(avail.keys())].rename(columns=avail)
        st.dataframe(prof_display.round(2), use_container_width=True, hide_index=True)
        
        # Heatmap
        st.markdown("---")
        fig_heat = _make_heatmap(profiles)
        st.plotly_chart(fig_heat, use_container_width=True)

        # Genre breakdown per cluster
        st.markdown('<div class="section-header">Primary Genre Mix per Cluster</div>', unsafe_allow_html=True)
        if "primary_genre" in seg_df.columns:
            genre_mix = (
                seg_df.groupby(["cluster_label", "primary_genre"])
                .size()
                .reset_index(name="count")
            )
            top_genres = (
                genre_mix.groupby("primary_genre")["count"].sum()
                .nlargest(10).index.tolist()
            )
            genre_mix = genre_mix[genre_mix["primary_genre"].isin(top_genres)]

            fig_genre = px.bar(
                genre_mix, x="cluster_label", y="count", color="primary_genre",
                barmode="stack",
                template="plotly_dark",
                color_discrete_sequence=ACCENT_COLORS,
                labels={"cluster_label": "Segment", "count": "Game Count", "primary_genre": "Genre"},
                title="Genre Composition per Segment (Top 10 Genres)",
            )
            fig_genre.update_layout(paper_bgcolor=PLOTLY_PAPER_BG, plot_bgcolor=PLOTLY_BG_COLOR,
                                    font=dict(color="#e0e0e0", family="Inter, sans-serif"))
            st.plotly_chart(fig_genre, use_container_width=True)

        # Price tier breakdown per cluster
        if "price_tier" in seg_df.columns:
            st.markdown('<div class="section-header">Price Tier Mix per Cluster</div>', unsafe_allow_html=True)
            tier_mix = (
                seg_df.groupby(["cluster_label", "price_tier"])
                .size()
                .reset_index(name="count")
            )
            fig_tier = px.bar(
                tier_mix, x="cluster_label", y="count", color="price_tier",
                barmode="stack",
                template="plotly_dark",
                color_discrete_sequence=ACCENT_COLORS,
                labels={"cluster_label": "Segment", "count": "Game Count", "price_tier": "Tier"},
                title="Price Tier Composition per Segment",
            )
            fig_tier.update_layout(paper_bgcolor=PLOTLY_PAPER_BG, plot_bgcolor=PLOTLY_BG_COLOR,
                                   font=dict(color="#e0e0e0", family="Inter, sans-serif"))
            st.plotly_chart(fig_tier, use_container_width=True)

    # ── TAB 3: Radar chart ────────────────────────────────────────────────────
    with tab_radar:
        st.markdown('<div class="section-header">Cluster Radar — Feature Comparison</div>', unsafe_allow_html=True)
        fig_radar = _make_radar(profiles)
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("""
        <div class="info-box">
            Each axis is independently normalised (0 = minimum, 1 = maximum across all clusters).
            This makes shapes comparable regardless of feature scale differences.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 4: Game lookup ────────────────────────────────────────────────────
    with tab_lookup:
        st.markdown('<div class="section-header">Find Which Segment a Game Belongs To</div>', unsafe_allow_html=True)
        search = st.text_input("Search game name", placeholder="e.g. Hollow Knight, Terraria…")

        if search:
            matches = seg_df[seg_df["name"].str.contains(search, case=False, na=False)]
            if matches.empty:
                st.warning("No games found matching that search term.")
            else:
                selected_name = st.selectbox("Select game", matches["name"].tolist())
                game_row = matches[matches["name"] == selected_name].iloc[0]

                seg_label = game_row.get("cluster_label", "Unknown")
                seg_id    = game_row.get("cluster_id", -1)

                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:1.3rem; font-weight:700; color:#e0e0e0;">{selected_name}</div>
                    <div style="color:#7C3AED; font-size:1.1rem; margin-top:6px;">
                        {seg_label}
                    </div>
                    <div style="color:#8b949e; font-size:0.9rem; margin-top:4px;">
                        Price: ${game_row.get('price', 0):.2f} ·
                        Review: {game_row['review_score_pct']*100:.1f}% ·
                        Est. Owners: {int(game_row.get('owners_mid', 0)):,} ·
                        Genre: {game_row.get('primary_genre', 'N/A')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Show other games in same cluster
                st.markdown(f'<div class="section-header">Other Games in {seg_label}</div>', unsafe_allow_html=True)
                cluster_peers = seg_df[
                    (seg_df["cluster_id"] == seg_id) &
                    (seg_df["name"] != selected_name)
                ].sample(min(20, len(seg_df[seg_df["cluster_id"] == seg_id]) - 1), random_state=42)

                peer_cols = ["name", "price", "review_score_pct", "owners_mid",
                             "primary_genre", "price_tier", "peak_ccu"]
                avail_peer = [c for c in peer_cols if c in cluster_peers.columns]
                st.dataframe(
                    cluster_peers[avail_peer].sort_values("owners_mid", ascending=False).round(3),
                    use_container_width=True, hide_index=True
                )
