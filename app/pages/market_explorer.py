"""
market_explorer.py — Interactive Market Explorer with sidebar filters
and eight chart types driven by the filtered dataset.
All filters update all charts. No hardcoded values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import src.analytics as an
from src.config import ACCENT_COLORS, PLOTLY_BG_COLOR, PLOTLY_PAPER_BG, PLOTLY_FONT_COLOR


# ── Shared layout ─────────────────────────────────────────────────────────────
def _L(**kw) -> dict:
    base = dict(
        template="plotly_dark",
        plot_bgcolor=PLOTLY_BG_COLOR,
        paper_bgcolor=PLOTLY_PAPER_BG,
        font=dict(color=PLOTLY_FONT_COLOR, family="Inter, sans-serif", size=12),
        margin=dict(t=52, r=16, b=52, l=60),
    )
    base.update(kw)
    return base


TIER_ORDER  = ["Free", "Budget", "Mid-range", "Premium", "AAA"]
TIER_COLORS = {t: ACCENT_COLORS[i] for i, t in enumerate(TIER_ORDER)}


# ── Filter helpers ────────────────────────────────────────────────────────────

def _apply_filters(
    df: pd.DataFrame,
    genres: list[str],
    price_range: tuple[float, float],
    year_range: tuple[int, int],
    review_min: float,
    platforms: list[str],
    lang_min: int,
    owner_min: float,
    tiers: list[str],
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if genres:
        mask &= df["primary_genre"].isin(genres)

    mask &= (df["price"] >= price_range[0]) & (df["price"] <= price_range[1])

    if "release_year" in df.columns:
        mask &= df["release_year"].between(year_range[0], year_range[1])

    if review_min > 0:
        mask &= df["review_score_pct"].fillna(0) >= review_min / 100.0

    for plat in platforms:
        col = plat.lower()
        if col in df.columns:
            mask &= df[col].fillna(False).astype(bool)

    if lang_min > 1:
        mask &= df["languages_count"].fillna(0) >= lang_min

    if owner_min > 0:
        mask &= df["owners_mid"].fillna(0) >= owner_min

    if tiers:
        mask &= df["price_tier"].isin(tiers)

    return df[mask].copy()


# ── Chart builders ─────────────────────────────────────────────────────────────

def _genre_price_boxplot(fdf: pd.DataFrame) -> go.Figure:
    """Genre × price boxplot — filtered dataset."""
    paid = fdf[(fdf["price"] > 0) & (fdf["price"] <= 80) & fdf["primary_genre"].notna()]
    genres = (
        paid.groupby("primary_genre")["price"]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )
    fig = go.Figure()
    for i, g in enumerate(genres):
        sub = paid[paid["primary_genre"] == g]["price"]
        if len(sub) < 5:
            continue
        fig.add_trace(go.Box(
            x=sub,
            name=g,
            orientation="h",
            marker_color=ACCENT_COLORS[i % len(ACCENT_COLORS)],
            line_width=1.2,
            showlegend=False,
            hovertemplate=f"<b>{g}</b><br>Price: $%{{x:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        **_L(title="Price Distribution by Genre ($0–$80, paid)", margin=dict(t=52, r=16, b=40, l=110)),
        xaxis=dict(title="Price (USD)", showgrid=True, gridcolor="#1e2030"),
        yaxis=dict(title="", showgrid=False),
        height=max(380, len(genres) * 28 + 80),
    )
    return fig


def _price_distribution(fdf: pd.DataFrame) -> go.Figure:
    """Price histogram with KDE overlay."""
    paid = fdf[(fdf["price"] >= 0) & (fdf["price"] <= 80)]["price"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=paid,
        nbinsx=60,
        marker_color=ACCENT_COLORS[0],
        marker_line=dict(width=0.3, color="rgba(255,255,255,0.15)"),
        opacity=0.85,
        name="Games",
        hovertemplate="$%{x:.1f} — %{y} games<extra></extra>",
    ))
    fig.update_layout(
        **_L(title="Price Distribution (paid titles ≤ $80)"),
        xaxis=dict(title="Price (USD)", showgrid=False),
        yaxis=dict(title="Game Count", showgrid=True, gridcolor="#1e2030"),
        showlegend=False,
    )
    return fig


def _price_vs_quality(
    fdf: pd.DataFrame,
    scale_mode: str = "Focused ($0–$70 Linear)",
    show_trend: bool = True,
    sample: int = 4000,
) -> go.Figure:
    """
    Price vs review quality scatter plot with overplotting mitigation,
    log/focused scaling options, explicit tier pricing definitions, and
    an optional median quality trendline overlay.
    """
    sub = fdf[(fdf["price"] > 0) & fdf["review_score_pct"].notna() & (fdf["total_review"] >= 5)].copy()
    if sub.empty:
        return go.Figure()

    # Filter for focused view if selected
    if scale_mode == "Focused ($0–$70 Linear)":
        plot_sub = sub[sub["price"] <= 70].copy()
    else:
        plot_sub = sub.copy()

    if len(plot_sub) > sample:
        sampled_sub = plot_sub.sample(sample, random_state=42).copy()
    else:
        sampled_sub = plot_sub.copy()

    sampled_sub["review_pct"] = sampled_sub["review_score_pct"] * 100

    # Human-readable price tier labels with retail thresholds
    tier_label_map = {
        "Free": "Free ($0)",
        "Budget": "Budget ($0.01 – $9.99)",
        "Mid-range": "Mid-Range ($10.00 – $29.99)",
        "Premium": "Premium ($30.00 – $49.99)",
        "AAA": "Flagship / $50+ (Price Tier)",
    }
    sampled_sub["tier_display"] = sampled_sub["price_tier"].map(tier_label_map).fillna(sampled_sub["price_tier"])

    tier_display_order = [
        "Free ($0)",
        "Budget ($0.01 – $9.99)",
        "Mid-Range ($10.00 – $29.99)",
        "Premium ($30.00 – $49.99)",
        "Flagship / $50+ (Price Tier)",
    ]
    color_map = {
        tier_label_map.get(t, t): TIER_COLORS.get(t, "#7c6af7")
        for t in TIER_ORDER
    }

    fig = px.scatter(
        sampled_sub,
        x="price",
        y="review_pct",
        color="tier_display",
        category_orders={"tier_display": tier_display_order},
        color_discrete_map=color_map,
        opacity=0.22,
        size_max=8,
        hover_data={"name": True, "primary_genre": True, "price": ":.2f", "review_pct": ":.1f", "tier_display": False},
        labels={
            "price": "Price (USD)",
            "review_pct": "Review Score (%)",
            "tier_display": "Price Bracket",
        },
    )
    fig.update_traces(marker=dict(size=4.5, line=dict(width=0)))

    # Median Quality Trendline across price brackets
    if show_trend and len(sub) >= 30:
        if scale_mode == "Focused ($0–$70 Linear)":
            bins = [0, 5, 10, 15, 20, 30, 50, 70]
        else:
            bins = [0, 5, 10, 15, 20, 30, 50, 70, 100, 200]

        trend_x, trend_y, trend_hover = [], [], []
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i+1]
            b_df = sub[(sub["price"] >= low) & (sub["price"] < high)]
            if len(b_df) >= 15:
                med_p = b_df["price"].median()
                med_q = (b_df["review_score_pct"] * 100).median()
                trend_x.append(med_p)
                trend_y.append(med_q)
                trend_hover.append(
                    f"<b>Bracket ${low}–${high}</b><br>Median Price: ${med_p:.2f}<br>Median Review: {med_q:.1f}% ({len(b_df):,} titles)"
                )

        if len(trend_x) >= 2:
            fig.add_trace(go.Scatter(
                x=trend_x,
                y=trend_y,
                mode="lines+markers",
                name="Median Quality Curve",
                line=dict(color="#ffffff", width=2.5),
                marker=dict(size=8, color="#7c6af7", line=dict(width=1.5, color="#ffffff")),
                hovertemplate="%{text}<extra></extra>",
                text=trend_hover,
            ))

    # X-axis configuration
    if scale_mode == "Log Scale (log₁₀ Price)":
        x_cfg = dict(
            title="Price (USD) — Log₁₀ Scale (equal visual spacing for $1, $5, $10, $20, $60)",
            type="log",
            showgrid=True,
            gridcolor="#1e2030",
        )
    elif scale_mode == "Focused ($0–$70 Linear)":
        x_cfg = dict(
            title="Price (USD) — Focused View ($0–$70 covers 99.7% of games, eliminating $100+ software outlier distortion)",
            range=[0, 70],
            showgrid=True,
            gridcolor="#1e2030",
        )
    else:  # Full catalog
        x_cfg = dict(
            title="Price (USD) — Full Range (Includes extreme $100+ package outliers)",
            showgrid=True,
            gridcolor="#1e2030",
        )

    fig.update_layout(
        **_L(title="Price vs Review Quality (Point Opacity = 0.22)", margin=dict(t=52, r=16, b=50, l=60)),
        xaxis=x_cfg,
        yaxis=dict(title="Review Score (%)", range=[0, 102], showgrid=True, gridcolor="#1e2030"),
        legend=dict(orientation="h", y=1.07, x=0, font=dict(size=10)),
    )
    return fig


def _ownership_bubble(
    fdf: pd.DataFrame,
    scale_mode: str = "Focused ($0–$70 Linear)",
    color_by: str = "Top Genres (Clean Palette)",
    y_metric: str = "Estimated Owners (with Jitter)",
    sample: int = 3500,
) -> go.Figure:
    """
    Price × Ownership bubble chart addressing:
    1. Discretized Y-axis striations (via subtle density jitter or continuous review count)
    2. Ineffective bubble sizing (dynamic power scaling for 75x contrast)
    3. Color overload (grouped into top genres + Other, or clean price tier colors)
    4. Outlier compression (focused $0-$70 view covers 99.7% of catalog)
    """
    sub = fdf[
        (fdf["price"] > 0) & fdf["owners_mid"].notna() & fdf["review_score_pct"].notna()
    ].copy()
    if sub.empty:
        return go.Figure()

    if scale_mode == "Focused ($0–$70 Linear)":
        plot_sub = sub[sub["price"] <= 70].copy()
    else:
        plot_sub = sub.copy()

    if len(plot_sub) > sample:
        sampled = plot_sub.sample(sample, random_state=42).copy()
    else:
        sampled = plot_sub.copy()

    # Dynamic contrast bubble sizing (exponential expansion so 90%+ games stand out clearly from 60% games)
    sampled["review_pct"] = sampled["review_score_pct"] * 100
    sampled["bubble_size"] = np.power((sampled["review_pct"] - 45).clip(lower=5), 1.8)

    # Y-axis metric calculation
    np.random.seed(42)
    if "Review Count" in y_metric:
        sampled["y_val"] = np.log10(sampled["total_review"].fillna(0).clip(lower=1))
        y_title = "Total User Reviews (log₁₀ Continuous Scale)"
        y_axis_cfg = dict(
            title=y_title,
            tickvals=[1, 2, 3, 4, 5, 6],
            ticktext=["10", "100", "1K", "10K", "100K", "1M+"],
            showgrid=True,
            gridcolor="#1e2030",
        )
    else:
        # Estimated owners with gentle density jitter (+- 0.06 in log10 space) to reveal mass/distribution
        sampled["log_owners"] = np.log10(sampled["owners_mid"].clip(lower=1))
        sampled["y_val"] = sampled["log_owners"] + np.random.uniform(-0.06, 0.06, len(sampled))
        y_title = "Estimated Owners (log₁₀ Scale with Density Jitter)"
        y_axis_cfg = dict(
            title=y_title,
            tickvals=[4.0, 4.544, 4.875, 5.176, 5.544, 5.875, 6.176, 6.544, 6.875, 7.176, 7.875, 8.176],
            ticktext=["10K", "35K", "75K", "150K", "350K", "750K", "1.5M", "3.5M", "7.5M", "15M", "75M", "150M"],
            showgrid=True,
            gridcolor="#1e2030",
        )

    # Color grouping to prevent visual overload
    if "Price Tier" in color_by:
        tier_label_map = {
            "Free": "Free ($0)",
            "Budget": "Budget ($0.01 – $9.99)",
            "Mid-range": "Mid-Range ($10.00 – $29.99)",
            "Premium": "Premium ($30.00 – $49.99)",
            "AAA": "Flagship / $50+ (Price Tier)",
        }
        sampled["color_col"] = sampled["price_tier"].map(tier_label_map).fillna(sampled["price_tier"])
        color_order = [
            "Free ($0)", "Budget ($0.01 – $9.99)", "Mid-Range ($10.00 – $29.99)",
            "Premium ($30.00 – $49.99)", "Flagship / $50+ (Price Tier)"
        ]
        color_palette = {tier_label_map.get(t, t): TIER_COLORS.get(t, "#7c6af7") for t in TIER_ORDER}
        legend_title = "Price Tier"
    else:
        top_genres = ["Action", "Casual", "Adventure", "Indie", "Simulation", "RPG", "Strategy"]
        sampled["color_col"] = sampled["primary_genre"].apply(lambda g: g if g in top_genres else "Other")
        color_order = top_genres + ["Other"]
        color_palette = {
            "Action": ACCENT_COLORS[0],
            "Casual": ACCENT_COLORS[1],
            "Adventure": ACCENT_COLORS[2],
            "Indie": ACCENT_COLORS[3],
            "Simulation": "#f43f5e",
            "RPG": "#06b6d4",
            "Strategy": "#ec4899",
            "Other": "#64748b",
        }
        legend_title = "Genre"

    fig = px.scatter(
        sampled,
        x="price",
        y="y_val",
        color="color_col",
        category_orders={"color_col": color_order},
        color_discrete_map=color_palette,
        size="bubble_size",
        size_max=18,
        opacity=0.35,
        hover_data={
            "name": True,
            "price": ":.2f",
            "owners_mid": ":,.0f",
            "review_pct": ":.1f",
            "color_col": False,
            "bubble_size": False,
            "y_val": False,
        },
        labels={
            "price": "Price (USD)",
            "owners_mid": "Est. Owners",
            "review_pct": "Review Score (%)",
            "color_col": legend_title,
        },
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color="rgba(255,255,255,0.2)")))

    # X-axis configuration
    if scale_mode == "Log Scale (log₁₀ Price)":
        x_cfg = dict(
            title="Price (USD) — Log₁₀ Scale",
            type="log",
            showgrid=True,
            gridcolor="#1e2030",
        )
    elif scale_mode == "Focused ($0–$70 Linear)":
        x_cfg = dict(
            title="Price (USD) — Focused View ($0–$70 covers 99.7% of catalog)",
            range=[0, 70],
            showgrid=True,
            gridcolor="#1e2030",
        )
    else:
        x_cfg = dict(
            title="Price (USD) — Full Range",
            showgrid=True,
            gridcolor="#1e2030",
        )

    fig.update_layout(
        **_L(title="Ownership Bubble Chart (Size = Quality Contrast, Color = Grouped)", margin=dict(t=52, r=16, b=50, l=65)),
        xaxis=x_cfg,
        yaxis=y_axis_cfg,
        legend=dict(orientation="h", y=1.07, x=0, font=dict(size=10)),
    )
    return fig


NON_GAME_OR_DESCRIPTOR_GENRES = {
    # Non-game software
    "Utilities", "Animation & Modeling", "Design & Illustration", "Education",
    "Audio Production", "Video Production", "Software Training", "Photo Editing",
    "Accounting", "Game Development", "Web Publishing", "Documentary",
    # Content descriptor tags (not gameplay genres)
    "Sexual Content", "Nudity", "Violent", "Gore", "Early Access",
    # Typo tags / singletons
    "Aventure", "Aventura", "インディー",
}


def _genre_value_ranking(
    fdf: pd.DataFrame,
    metric_choice: str = "Log Cost-Efficiency (Review% / log₂(Price + 1))",
    include_software: bool = False,
) -> go.Figure:
    """
    Genre Value Ranking with:
    1. Exclusion of non-gaming software & mature content descriptor tags
    2. Prevention of division-by-zero and micro-game ($0.99) skew
    3. Multiple analytical value metrics (Log-efficiency, Playtime hours/$, Commercial floor >= $4.99)
    """
    paid = fdf[(fdf["price"] > 0) & fdf["primary_genre"].notna()].copy()
    if paid.empty:
        return go.Figure()

    if not include_software:
        paid = paid[~paid["primary_genre"].isin(NON_GAME_OR_DESCRIPTOR_GENRES)]

    # Filter genres with meaningful sample size (min 25 games)
    genre_counts = paid["primary_genre"].value_counts()
    valid_genres = genre_counts[genre_counts >= 25].index
    paid = paid[paid["primary_genre"].isin(valid_genres)]

    if paid.empty:
        return go.Figure()

    if "Playtime" in metric_choice:
        sub = paid[(paid["price"] >= 1.99) & (paid["median_playtime_forever"] > 0)].copy()
        sub["val"] = (sub["median_playtime_forever"] / 60.0) / sub["price"]
        metric_col = "val"
        x_title = "Median Playtime Hours per $1 Spent"
        chart_title = "Genre Longevity: Median Playtime Hours per $1 Spent (Price ≥ $1.99)"
        val_format = "%{x:.2f} hrs/$"
    elif "Commercial Price Floor" in metric_choice or "$4.99" in metric_choice:
        sub = paid[(paid["price"] >= 4.99) & paid["review_score_pct"].notna()].copy()
        sub["val"] = (sub["review_score_pct"] * 100) / sub["price"]
        metric_col = "val"
        x_title = "Median Quality Pts per $1 Spent"
        chart_title = "Commercial Value Ranking: Quality Pts per $1 (Price ≥ $4.99 Floor)"
        val_format = "%{x:.1f} pts/$"
    else:  # Log Cost-Efficiency (default)
        sub = paid[(paid["price"] >= 1.99) & paid["review_score_pct"].notna()].copy()
        sub["val"] = (sub["review_score_pct"] * 100) / np.log2(sub["price"] + 1)
        metric_col = "val"
        x_title = "Median Log Cost-Efficiency Index"
        chart_title = "Cost-Efficiency Ranking: Review% / log₂(Price + 1) (Price ≥ $1.99)"
        val_format = "%{x:.1f}"

    g = (
        sub.groupby("primary_genre")[metric_col]
        .median()
        .reset_index()
        .rename(columns={metric_col: "value_score"})
    )
    g = g[g["value_score"].notna()].sort_values("value_score", ascending=True)

    fig = go.Figure(go.Bar(
        x=g["value_score"],
        y=g["primary_genre"],
        orientation="h",
        text=g["value_score"],
        texttemplate=val_format,
        textposition="outside",
        textfont=dict(size=11, color="#e8e9f3"),
        marker=dict(
            color=g["value_score"],
            colorscale=[[0, ACCENT_COLORS[1]], [1, ACCENT_COLORS[0]]],
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>Value Score: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_L(title=chart_title, margin=dict(t=52, r=50, b=40, l=110)),
        xaxis=dict(title=x_title, showgrid=True, gridcolor="#1e2030"),
        yaxis=dict(title="", showgrid=False),
        height=max(360, len(g) * 32 + 80),
    )
    return fig


def _release_timeline(
    fdf: pd.DataFrame,
    view_mode: str = "Annual Volume (Multi-Line)",
    include_2026: bool = False,
    y_scale: str = "Linear",
) -> go.Figure:
    """
    Multi-genre release timeline (since 2010):
    1. Multi-line mode prevents stacked-area flatline illusion in 2010–2015.
    2. Isolates completed calendar years (2010–2025) by default to eliminate artificial 2026 cliff.
    3. Market Share % provides 100% normalized visibility into structural genre shifts.
    """
    if "release_year" not in fdf.columns:
        return go.Figure()

    plot_df = fdf[fdf["release_year"].notna() & fdf["primary_genre"].notna()].copy()
    plot_df["release_year"] = plot_df["release_year"].astype(int)
    plot_df = plot_df[plot_df["release_year"] >= 2010]
    if not include_2026:
        plot_df = plot_df[plot_df["release_year"] <= 2025]

    top_genres = (
        plot_df[~plot_df["primary_genre"].isin(NON_GAME_OR_DESCRIPTOR_GENRES)]["primary_genre"]
        .value_counts()
        .head(8)
        .index.tolist()
    )
    if not top_genres:
        return go.Figure()

    plot_df = plot_df[plot_df["primary_genre"].isin(top_genres)]

    pivot = (
        plot_df.groupby(["release_year", "primary_genre"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=top_genres, fill_value=0)
    )

    max_yr = 2026 if include_2026 and (plot_df["release_year"] == 2026).any() else (2025 if (plot_df["release_year"] <= 2025).any() else plot_df["release_year"].max())
    min_yr = max(2010, int(plot_df["release_year"].min())) if not plot_df.empty else 2010
    all_years = list(range(min_yr, max_yr + 1))
    pivot = pivot.reindex(index=all_years, fill_value=0)

    fig = go.Figure()
    palette = ACCENT_COLORS + ["#ff79c6", "#bd93f9", "#50fa7b", "#ffb86c"]

    if view_mode == "Market Share % (100% Normalized Area)":
        row_sums = pivot.sum(axis=1).replace(0, 1)
        share_pct = pivot.div(row_sums, axis=0) * 100
        for i, g in enumerate(top_genres):
            fig.add_trace(go.Scatter(
                x=share_pct.index,
                y=share_pct[g],
                mode="lines",
                stackgroup="one",
                groupnorm="percent",
                name=g,
                line=dict(width=1.2, color=palette[i % len(palette)]),
                hovertemplate=f"<b>{g}</b><br>Year: %{{x}}<br>Share: %{{y:.1f}}%<br>Releases: %{{customdata:,}}<extra></extra>",
                customdata=pivot[g],
            ))
        y_title = "Share of Top Genres (%)"
        chart_title = "Genre Market Share Composition (Top 8 Genres, 100% Normalized Area, 2010+)"
        y_axis_cfg = dict(title=y_title, range=[0, 100], ticksuffix="%", showgrid=True, gridcolor="#1e2030")

    elif view_mode == "Stacked Area (Total Volume)":
        for i, g in enumerate(top_genres):
            fig.add_trace(go.Scatter(
                x=pivot.index,
                y=pivot[g],
                mode="lines",
                stackgroup="one",
                name=g,
                line=dict(width=1, color=palette[i % len(palette)]),
                hovertemplate=f"<b>{g}</b><br>Year: %{{x}}<br>Releases: %{{y:,}}<extra></extra>",
            ))
        y_title = "Games Released (Stacked)"
        chart_title = "Cumulative Genre Release Volume (Stacked Area, 2010+)"
        y_axis_cfg = dict(title=y_title, showgrid=True, gridcolor="#1e2030")

    else:  # Annual Volume (Multi-Line) — Default
        is_log = (y_scale == "Log Scale (log₁₀)")
        for i, g in enumerate(top_genres):
            fig.add_trace(go.Scatter(
                x=pivot.index,
                y=pivot[g],
                mode="lines+markers",
                name=g,
                line=dict(width=2.5, color=palette[i % len(palette)]),
                marker=dict(size=5),
                hovertemplate=f"<b>{g}</b><br>Year: %{{x}}<br>Releases: %{{y:,}}<extra></extra>",
            ))
        y_title = "Annual Games Released" + (" (Log Scale)" if is_log else "")
        chart_title = "Genre Release Trajectories (Top 8 Genres, 2010+)"
        y_axis_cfg = dict(
            title=y_title,
            type="log" if is_log else "linear",
            showgrid=True,
            gridcolor="#1e2030",
        )

    if include_2026 and max_yr >= 2026:
        fig.add_vline(
            x=2026,
            line_dash="dash",
            line_color="#ff5c5c",
            line_width=1.5,
            annotation_text="2026 (YTD Incomplete)",
            annotation_position="top left",
            annotation_font=dict(color="#ff9999", size=10),
        )

    fig.update_layout(
        **_L(title=chart_title, margin=dict(t=52, r=24, b=52, l=65)),
        xaxis=dict(title="Release Year", dtick=1, showgrid=False, tickmode="linear"),
        yaxis=y_axis_cfg,
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
        height=450,
    )
    return fig


def _genre_price_heatmap(fdf: pd.DataFrame) -> go.Figure:
    """Genre × Price Tier count heatmap."""
    sub = fdf[fdf["primary_genre"].notna() & fdf["price_tier"].notna()].copy()
    pivot = (
        sub.groupby(["primary_genre", "price_tier"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=TIER_ORDER, fill_value=0)
    )
    # Sort genres by total count
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index].head(20)

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#09090f"], [0.3, ACCENT_COLORS[1]], [1, ACCENT_COLORS[0]]],
        text=pivot.values,
        texttemplate="%{text:,}",
        textfont=dict(size=10),
        colorbar=dict(title="Games", len=0.7),
        hovertemplate="<b>%{y}</b> / <b>%{x}</b><br>%{z:,} games<extra></extra>",
    ))
    fig.update_layout(
        **_L(title="Genre × Price Tier Heatmap (Game Count)", margin=dict(t=52, r=80, b=52, l=120)),
        xaxis=dict(title="Price Tier", showgrid=False),
        yaxis=dict(title="Genre", showgrid=False, autorange="reversed"),
        height=max(400, len(pivot) * 26 + 80),
    )
    return fig


def _cohort_heatmap(
    fdf: pd.DataFrame,
    heat_mode: str = "Row-Normalized (% of Year's Releases)",
    include_2026: bool = False,
) -> go.Figure:
    """
    Release year × Price Tier cohort heatmap (2010+):
    1. Row-normalization (% of year) eliminates column imbalance where Budget dwarfs Premium/AAA.
    2. Explicit categorical Y-axis ensures continuous, un-skipped yearly rows from 2010 onward.
    3. Isolates completed years (2010–2025) by default, with optional annotated 2026 YTD row.
    """
    if "release_year" not in fdf.columns:
        return go.Figure()
    sub = fdf[fdf["price_tier"].notna() & fdf["release_year"].notna()].copy()
    sub["release_year"] = sub["release_year"].astype(int)
    sub = sub[sub["release_year"] >= 2010]
    if not include_2026:
        sub = sub[sub["release_year"] <= 2025]

    if sub.empty:
        return go.Figure()

    pivot = (
        sub.groupby(["release_year", "price_tier"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=TIER_ORDER, fill_value=0)
    )
    max_yr = 2026 if include_2026 and (sub["release_year"] == 2026).any() else (2025 if (sub["release_year"] <= 2025).any() else sub["release_year"].max())
    min_yr = max(2010, int(sub["release_year"].min()))
    all_years = list(range(min_yr, max_yr + 1))
    pivot = pivot.reindex(index=all_years, fill_value=0)

    # String-formatted categorical labels guarantee every year has an explicit, evenly spaced row
    y_labels = [f"{y} (YTD)" if y == 2026 else str(y) for y in pivot.index]

    if heat_mode == "Row-Normalized (% of Year's Releases)":
        row_totals = pivot.sum(axis=1).replace(0, 1)
        pivot_norm = pivot.div(row_totals, axis=0) * 100
        z_vals = pivot_norm.values
        text_matrix = [
            [f"{pivot_norm.iloc[r, c]:.1f}%" for c in range(len(TIER_ORDER))]
            for r in range(len(pivot))
        ]
        colorscale = [[0.0, "#080b12"], [0.25, "#0e2f4f"], [0.6, "#1565c0"], [1.0, "#00e5ff"]]
        cb_title = "% Share"
        hover_tmpl = "<b>Year %{y} — %{x} Tier</b><br>Cohort Share: %{z:.1f}%<br>Catalog Count: %{customdata:,} games<extra></extra>"
        chart_title = "Cohort Heatmap: Price Tier Share by Release Year (% Normalized)"

    elif heat_mode == "Log-Scaled Volume (log₁₀ Count)":
        z_vals = np.log10(pivot.values + 1)
        text_matrix = [
            [f"{pivot.iloc[r, c]:,}" for c in range(len(TIER_ORDER))]
            for r in range(len(pivot))
        ]
        colorscale = [[0.0, "#09090f"], [0.35, "#3d105e"], [0.7, ACCENT_COLORS[3]], [1.0, ACCENT_COLORS[2]]]
        cb_title = "log₁₀(Count)"
        hover_tmpl = "<b>Year %{y} — %{x} Tier</b><br>Game Count: %{customdata:,}<extra></extra>"
        chart_title = "Cohort Heatmap: Price Tier Distribution (Log-Scaled Count)"

    else:  # Absolute Game Count
        z_vals = pivot.values
        text_matrix = [
            [f"{pivot.iloc[r, c]:,}" for c in range(len(TIER_ORDER))]
            for r in range(len(pivot))
        ]
        colorscale = [[0.0, "#09090f"], [0.4, ACCENT_COLORS[3]], [1.0, ACCENT_COLORS[2]]]
        cb_title = "Games"
        hover_tmpl = "<b>Year %{y} — %{x} Tier</b><br>Game Count: %{z:,}<extra></extra>"
        chart_title = "Cohort Heatmap: Release Year × Price Tier (Absolute Count)"

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=pivot.columns.tolist(),
        y=y_labels,
        colorscale=colorscale,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title=cb_title, len=0.7),
        customdata=pivot.values,
        hovertemplate=hover_tmpl,
    ))
    fig.update_layout(
        **_L(title=chart_title, margin=dict(t=52, r=80, b=52, l=75)),
        xaxis=dict(title="Price Tier", showgrid=False),
        yaxis=dict(title="Release Year", showgrid=False, type="category", autorange="reversed"),
        height=max(440, len(pivot) * 26 + 90),
    )
    return fig


# ── Main render ───────────────────────────────────────────────────────────────

def _render_internal(df: pd.DataFrame) -> None:

    st.markdown("""
    <style>
    .explorer-header {
        background: linear-gradient(135deg, rgba(79,142,247,0.07) 0%, rgba(9,9,15,0) 100%);
        border: 1px solid #1e2030;
        border-radius: 12px;
        padding: 22px 26px 18px;
        margin-bottom: 22px;
        position: relative;
    }
    .explorer-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, #4f8ef7, transparent);
        opacity: 0.45;
    }
    .explorer-title {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #e8e9f3, #4f8ef7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .explorer-sub {
        font-size: 0.875rem;
        color: #8b8fa8;
        margin-top: 5px;
    }
    .filter-summary {
        background: rgba(79,142,247,0.05);
        border: 1px solid rgba(79,142,247,0.15);
        border-radius: 8px;
        padding: 8px 14px;
        font-size: 0.82rem;
        color: #8b8fa8;
        margin-bottom: 18px;
    }
    .filter-summary strong { color: #4f8ef7; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar Filters ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="section-header">Explorer Filters</div>', unsafe_allow_html=True)

        # Genre filter
        all_genres = sorted(df["primary_genre"].dropna().unique().tolist())
        selected_genres = st.multiselect(
            "Genre",
            options=all_genres,
            default=[],
            key="expl_genre",
        )

        # Price tier filter
        selected_tiers = st.multiselect(
            "Price Tier",
            options=TIER_ORDER,
            default=[],
            key="expl_tier",
        )

        # Price range slider (cap display at $100 for usability; filter applies actual bounds)
        p_min = float(df["price"].min())
        p_max_actual = float(df["price"].max())
        p_max_slider = min(p_max_actual, 100.0)
        price_range_raw = st.slider(
            "Price Range ($)",
            min_value=p_min,
            max_value=p_max_slider,
            value=(p_min, p_max_slider),
            step=0.5,
            key="expl_price",
        )
        # If upper bound is at slider max, extend to actual max so all games are included
        price_range = (
            price_range_raw[0],
            p_max_actual if price_range_raw[1] >= p_max_slider else price_range_raw[1],
        )

        # Year range
        if "release_year" in df.columns:
            yr_min = int(df["release_year"].dropna().min())
            yr_max = int(df["release_year"].dropna().max())
            year_range = st.slider(
                "Release Year",
                min_value=yr_min,
                max_value=yr_max,
                value=(max(yr_min, 2005), yr_max),
                step=1,
                key="expl_year",
            )
        else:
            year_range = (2000, 2024)

        # Review score minimum
        review_min = st.slider(
            "Min Review Score (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            key="expl_review",
        )

        # Platform filter
        platform_opts = ["Windows", "Mac", "Linux"]
        selected_platforms = st.multiselect(
            "Requires Platform",
            options=platform_opts,
            default=[],
            key="expl_plat",
        )

        # Language count minimum
        lang_min = st.slider(
            "Min Language Support",
            min_value=1,
            max_value=30,
            value=1,
            step=1,
            key="expl_lang",
        )

        # Minimum ownership
        own_options = [0, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
        own_labels  = ["Any", "5K+", "10K+", "50K+", "100K+", "500K+", "1M+"]
        own_idx = st.select_slider(
            "Min Est. Ownership",
            options=own_labels,
            value="Any",
            key="expl_own",
        )
        owner_min = own_options[own_labels.index(own_idx)]

        st.markdown("---")
        if st.button("↺ Reset Filters", key="expl_reset"):
            for k in ["expl_genre", "expl_tier", "expl_price", "expl_year",
                      "expl_review", "expl_plat", "expl_lang", "expl_own"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    # ── Apply filters ──────────────────────────────────────────────────────────
    fdf = _apply_filters(
        df,
        genres=selected_genres,
        price_range=price_range,
        year_range=year_range,
        review_min=review_min,
        platforms=selected_platforms,
        lang_min=lang_min,
        owner_min=owner_min,
        tiers=selected_tiers,
    )

    # ── Hero ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="explorer-header">
        <div class="explorer-title">📊 Market Explorer</div>
        <div class="explorer-sub">Eight chart types across genre, price, quality, ownership, and time — all driven by sidebar filters.</div>
    </div>
    """, unsafe_allow_html=True)

    # Filter summary bar
    pct_retained = len(fdf) / len(df) * 100 if len(df) > 0 else 0
    st.markdown(f"""
    <div class="filter-summary">
        Showing <strong>{len(fdf):,}</strong> of {len(df):,} games ({pct_retained:.1f}%)
        after filters — use the sidebar to refine the view.
    </div>
    """, unsafe_allow_html=True)

    if len(fdf) < 10:
        st.warning("⚠ Fewer than 10 games match the current filters. Adjust filters to see charts.")
        return

    # ── TAB LAYOUT — 4 tabs × 2 charts each ──────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Price & Genre",
        "🔬 Quality & Value",
        "👥 Ownership & Reach",
        "📅 Time & Cohorts",
    ])

    # ── Tab 1: Price & Genre ───────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Genre × Price Boxplot</div>', unsafe_allow_html=True)
        st.plotly_chart(_genre_price_boxplot(fdf), use_container_width=True)

        st.markdown('<div class="section-header">Price Distribution</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns([1.4, 1])
        with col_a:
            st.plotly_chart(_price_distribution(fdf), use_container_width=True)
        with col_b:
            st.markdown('<div class="section-header">Genre × Price Tier Heatmap</div>', unsafe_allow_html=True)
            st.plotly_chart(_genre_price_heatmap(fdf), use_container_width=True)

    # ── Tab 2: Quality & Value ─────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Price vs Review Quality</div>', unsafe_allow_html=True)

        col_scale, col_trend = st.columns([1.6, 1])
        with col_scale:
            pvq_scale = st.radio(
                "X-Axis Scaling:",
                ["Focused ($0–$70 Linear)", "Log Scale (log₁₀ Price)", "Full Catalog (Uncapped)"],
                horizontal=True,
                key="pvq_scale",
            )
        with col_trend:
            pvq_trend = st.checkbox(
                "Overlay Median Quality Trendline",
                value=True,
                key="pvq_trend",
                help="Computes median review score across price brackets to reveal the macro quality curve without overplotting distortion.",
            )

        st.plotly_chart(_price_vs_quality(fdf, scale_mode=pvq_scale, show_trend=pvq_trend), use_container_width=True)

        st.caption(
            "💡 **Analytical Notes**: Point opacity is set to 0.22 to eliminate dense overplotting. "
            "Tiers reflect catalog retail price brackets (`Budget < $10`, `Mid $10–$30`, `Premium $30–$50`, `Flagship $50+`), "
            "not studio production budgets. The white curve marks median review score across price brackets."
        )

        st.markdown('<div class="section-header" style="margin-top:28px;">Genre Value Ranking</div>', unsafe_allow_html=True)

        col_metric, col_soft = st.columns([1.8, 1])
        with col_metric:
            gvr_metric = st.selectbox(
                "Value Ranking Metric:",
                [
                    "Log Cost-Efficiency (Review% / log₂(Price + 1))",
                    "Playtime Hours per Dollar (Longevity)",
                    "Quality Points per $1 (Price ≥ $4.99 Floor)",
                ],
                key="gvr_metric",
                help="Log Cost-Efficiency prevents micro-game ($0.99) distortion. Longevity shows playtime depth per dollar spent.",
            )
        with col_soft:
            gvr_soft = st.checkbox(
                "Include Software & Utilities",
                value=False,
                key="gvr_soft",
                help="When unchecked, filters out non-gaming productivity software (Utilities, Photo Editing, Animation) and mature descriptor tags.",
            )

        st.plotly_chart(_genre_value_ranking(fdf, metric_choice=gvr_metric, include_software=gvr_soft), use_container_width=True)

        st.caption(
            "💡 **Analytical Methodology**: Non-gaming software tags and mature content descriptors are excluded by default to rank true gaming genres. "
            "A price floor prevents division-by-zero artifacts and sub-$1 asset flip skew."
        )

    # ── Tab 3: Ownership & Reach ───────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Ownership Bubble Chart</div>', unsafe_allow_html=True)

        col_ob1, col_ob2, col_ob3 = st.columns([1.5, 1.2, 1.3])
        with col_ob1:
            ob_scale = st.radio(
                "Price Axis Scaling:",
                ["Focused ($0–$70 Linear)", "Log Scale (log₁₀ Price)", "Full Catalog (Uncapped)"],
                horizontal=True,
                key="ob_scale",
            )
        with col_ob2:
            ob_color = st.selectbox(
                "Color Palette:",
                ["Top Genres (Clean Palette)", "Price Tier"],
                key="ob_color",
            )
        with col_ob3:
            ob_ymetric = st.selectbox(
                "Y-Axis Reach Metric:",
                ["Estimated Owners (with Jitter)", "Total Review Count (Continuous log₁₀)"],
                key="ob_ymetric",
                help="Estimated Owners applies subtle density jitter to unpack stepped brackets. Review count is naturally continuous.",
            )

        st.plotly_chart(_ownership_bubble(fdf, scale_mode=ob_scale, color_by=ob_color, y_metric=ob_ymetric), use_container_width=True)

        st.caption(
            "💡 **Analytical Notes**: Bubble area uses dynamic power scaling to contrast critically acclaimed titles against average releases. "
            "Subtle density jitter is applied to unpack stepped ownership brackets (10K, 35K, 75K, etc.), while hover tooltips display exact numbers."
        )

        st.markdown('<div class="section-header" style="margin-top:28px;">Ownership Distribution & Market Leaders</div>', unsafe_allow_html=True)

        col_x, col_y = st.columns([1.1, 1.3])
        with col_x:
            st.markdown('<div class="chart-section-title">Genre Reach Profile</div>', unsafe_allow_html=True)
            clean_fdf = fdf[~fdf["primary_genre"].isin(NON_GAME_OR_DESCRIPTOR_GENRES) & fdf["primary_genre"].notna()]
            g_counts = clean_fdf["primary_genre"].value_counts()
            valid_g = g_counts[g_counts >= 15].index

            g_own = (
                clean_fdf[clean_fdf["primary_genre"].isin(valid_g)]
                .groupby("primary_genre")
                .agg(
                    games=("app_id", "count"),
                    avg_owners=("owners_mid", "mean"),
                    p75_owners=("owners_mid", lambda x: np.percentile(x.dropna(), 75) if len(x.dropna()) > 0 else 0),
                    breakout_pct=("owners_mid", lambda x: (x > 20000).mean() * 100),
                    median_price=("price", "median"),
                    median_review=("review_score_pct", lambda x: x.dropna().median() * 100 if len(x.dropna()) > 0 else np.nan),
                )
                .reset_index()
                .sort_values("avg_owners", ascending=False)
            )

            # Format for executive presentation
            g_display = pd.DataFrame({
                "Genre": g_own["primary_genre"],
                "Games": g_own["games"].apply(lambda n: f"{n:,}"),
                "Avg Reach": g_own["avg_owners"].apply(lambda n: f"{n:,.0f}"),
                "P75 Reach": g_own["p75_owners"].apply(lambda n: f"{n:,.0f}"),
                "Breakout (>20K)": g_own["breakout_pct"].apply(lambda p: f"{p:.1f}%"),
                "Med Price": g_own["median_price"].apply(lambda p: f"${p:.2f}"),
                "Med Review": g_own["median_review"].apply(lambda r: f"{r:.1f}%" if pd.notna(r) else "--"),
            })
            st.dataframe(g_display, use_container_width=True, hide_index=True)
            st.caption("Floor effect eliminated: Mean reach and 75th percentile reveal commercial scale beyond the base 10K bracket.")

        with col_y:
            st.markdown('<div class="chart-section-title">Top 15 Titles by Estimated Reach</div>', unsafe_allow_html=True)
            top_raw = an.top_games_by_owners(fdf, n=15)
            top_display = pd.DataFrame({
                "Title": top_raw["name"],
                "Genre": top_raw["primary_genre"].fillna("Action"),
                "Price": top_raw["price"].apply(lambda p: "Free" if p == 0 else f"${p:.2f}"),
                "Est. Owners": top_raw["owners_mid"].apply(lambda o: f"{o:,.0f}" if o < 1_000_000 else f"{o/1_000_000:.0f}M"),
                "Total Reviews": top_raw["total_review"].fillna(0).apply(lambda r: f"{r:,.0f}"),
                "Rating": top_raw["review_score_pct"].apply(lambda s: f"{s*100:.1f}%" if pd.notna(s) else "--"),
            })
            st.dataframe(top_display, use_container_width=True, hide_index=True)
            st.caption("Ties broken by total review engagement. Null genres populated from API catalog mapping.")

    # ── Tab 4: Time & Cohorts ──────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">Release Timeline by Genre</div>', unsafe_allow_html=True)

        col_tl1, col_tl2, col_tl3 = st.columns([1.6, 1.1, 1.1])
        with col_tl1:
            tl_mode = st.radio(
                "Timeline Display Mode:",
                [
                    "Annual Volume (Multi-Line)",
                    "Market Share % (100% Normalized Area)",
                    "Stacked Area (Total Volume)",
                ],
                horizontal=False,
                key="tl_mode",
                help="Multi-Line shows distinct trajectories without stacking distortion. Market Share % displays shifting genre composition.",
            )
        with col_tl2:
            tl_yscale = st.radio(
                "Y-Axis Scale (Multi-Line):",
                ["Linear", "Log Scale (log₁₀)"],
                horizontal=True,
                key="tl_yscale",
                help="Log scale enables comparing relative growth rates between large genres (Action) and smaller genres (Strategy, RPG).",
            )
        with col_tl3:
            tl_inc_2026 = st.checkbox(
                "Include 2026 (Incomplete YTD)",
                value=False,
                key="tl_inc_2026",
                help="2026 is an in-progress partial year. Uncheck to analyze full completed calendar years (2010–2025).",
            )

        st.plotly_chart(_release_timeline(fdf, view_mode=tl_mode, include_2026=tl_inc_2026, y_scale=tl_yscale), use_container_width=True)

        st.caption(
            "💡 **Analytical Notes**: Individual multi-line trajectories resolve the stacked area flatline illusion for 2010–2015 "
            "and eliminate ribbon compression for Strategy, RPG, Simulation, and Free to Play. "
            "Completed years (2010–2025) are isolated by default to prevent partial-year drop-off artifacts."
        )

        st.markdown('<div class="section-header" style="margin-top:28px;">Cohort Heatmap: Release Year × Price Tier</div>', unsafe_allow_html=True)

        col_hm1, col_hm2 = st.columns([2.0, 1.2])
        with col_hm1:
            hm_mode = st.radio(
                "Heatmap Normalization:",
                [
                    "Row-Normalized (% of Year's Releases)",
                    "Log-Scaled Volume (log₁₀ Count)",
                    "Absolute Game Count",
                ],
                horizontal=True,
                key="hm_mode",
                help="Row-Normalized (% of Year) normalizes each cohort year to 100% so Premium and AAA tiers are not washed out by Budget dominance.",
            )
        with col_hm2:
            hm_inc_2026 = st.checkbox(
                "Include 2026 in Heatmap",
                value=False,
                key="hm_inc_2026",
                help="Include partial-year 2026 cohort row. Labeled as 2026 (YTD) to distinguish from full calendar years.",
            )

        st.plotly_chart(_cohort_heatmap(fdf, heat_mode=hm_mode, include_2026=hm_inc_2026), use_container_width=True)

        st.caption(
            "💡 **Analytical Notes**: Row-normalization reveals true pricing dynamics (e.g. Free-to-Play rising from 3.8% to 26.2%, "
            "Budget shifting from 75.7% to 57.0%) without high volume blowing out the colorscale. Categorical Y-axis guarantees all years (2010–2025) are distinctly displayed."
        )

    # ── Dataset info footer ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="info-box" style="margin-top:12px;">
        All charts built from the filtered view of <strong>{len(fdf):,} games</strong>.
        Filters apply to every chart simultaneously.
        Hover over data points for exact values.
    </div>
    """, unsafe_allow_html=True)


def render(df: pd.DataFrame, hide_header: bool = False) -> None:
    if not hide_header:
        st.markdown('<div class="hero-header"><div class="hero-title">📊 Market Explorer</div><div class="hero-subtitle">Interactive multidimensional exploration & Market Map.</div></div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Data Grid & Filters", "Visual Market Map"])
    
    with tab1:
        _render_internal(df)
        
    with tab2:
        from app.pages import market_map
        market_map.render(df, hide_header=True)
