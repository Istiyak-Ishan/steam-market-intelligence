"""
visualization.py — Reusable Plotly chart factory for the Steam Market Intelligence Platform.

Every function:
  1. Accepts a DataFrame/Series from the analytics layer (never raw data)
  2. Returns a Plotly Figure object
  3. Uses the platform dark theme consistently
  4. Answers a specific analytical question

No charts are built "just because they look impressive."
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import (
    PLOTLY_TEMPLATE, PLOTLY_BG_COLOR, PLOTLY_PAPER_BG,
    PLOTLY_FONT_COLOR, ACCENT_COLORS,
)


# ── Shared layout helper ──────────────────────────────────────────────────────
def _base_layout(**kwargs) -> dict:
    base = dict(
        template=PLOTLY_TEMPLATE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PLOTLY_FONT_COLOR, family="Rajdhani, sans-serif", size=13),
        margin=dict(t=60, r=20, b=60, l=60),
        hoverlabel=dict(bgcolor="rgba(10, 14, 23, 0.9)", font_size=13, font_family="JetBrains Mono"),
    )
    base.update(kwargs)
    return base


def _apply_theme(fig: go.Figure, title: str = "", **layout_kwargs) -> go.Figure:
    fig.update_layout(title=dict(text=title, font_size=18, font_family="Orbitron", font_color="#00F0FF", x=0.01), **_base_layout(**layout_kwargs))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1e3a5f", griddash="dot", zerolinecolor="#00F0FF", zerolinewidth=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1e3a5f", griddash="dot", zerolinecolor="#00F0FF", zerolinewidth=1)
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# ── DISTRIBUTION CHARTS ─────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def price_histogram(price_series: pd.Series, bins: int = 60) -> go.Figure:
    """Histogram of game prices filtered to ≤ $80."""
    fig = px.histogram(
        price_series.clip(upper=80),
        nbins=bins,
        labels={"value": "Price (USD)", "count": "Game Count"},
        color_discrete_sequence=[ACCENT_COLORS[0]],
    )
    fig.update_traces(marker_line_width=0.3, marker_line_color="rgba(255,255,255,0.2)")
    return _apply_theme(fig, "Distribution of Steam Game Prices (≤ $80)")


def review_score_histogram(score_series: pd.Series) -> go.Figure:
    """Histogram of review score percentages (0–100)."""
    fig = px.histogram(
        score_series * 100,
        nbins=50,
        labels={"value": "Review Score (%)", "count": "Game Count"},
        color_discrete_sequence=[ACCENT_COLORS[1]],
    )
    return _apply_theme(fig, "Distribution of Steam Review Scores")


def playtime_histogram(playtime_hours: pd.Series) -> go.Figure:
    """Histogram of average playtime in hours."""
    fig = px.histogram(
        playtime_hours,
        nbins=60,
        labels={"value": "Avg Playtime (hours)", "count": "Game Count"},
        color_discrete_sequence=[ACCENT_COLORS[2]],
    )
    return _apply_theme(fig, "Average Playtime Distribution (hours, capped at 200h)")


# ════════════════════════════════════════════════════════════════════════════════
# ── BAR CHARTS ─────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def genre_bar(df: pd.DataFrame, y_col: str, title: str, y_label: str = "") -> go.Figure:
    """Horizontal bar chart comparing genres on a single metric."""
    sorted_df = df.sort_values(y_col, ascending=True)
    fig = px.bar(
        sorted_df,
        x=y_col,
        y="primary_genre",
        orientation="h",
        color=y_col,
        color_continuous_scale=["#3B82F6", "#7C3AED"],
        labels={"primary_genre": "Genre", y_col: y_label or y_col},
    )
    fig.update_coloraxes(showscale=False)
    return _apply_theme(fig, title)


def price_tier_bar(tier_df: pd.DataFrame) -> go.Figure:
    """Bar chart of game count per price tier."""
    fig = px.bar(
        tier_df,
        x="price_tier",
        y="count",
        text="pct",
        color="price_tier",
        color_discrete_sequence=ACCENT_COLORS,
        labels={"price_tier": "Price Tier", "count": "Number of Games"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    return _apply_theme(fig, "Game Count by Price Tier")


def genre_grouped_bar(
    df: pd.DataFrame,
    metrics: list[str],
    genre_col: str = "primary_genre",
    title: str = "",
) -> go.Figure:
    """Grouped bar chart comparing multiple metrics across genres."""
    fig = go.Figure()
    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
        fig.add_trace(go.Bar(
            name=metric,
            x=df[genre_col],
            y=df[metric],
            marker_color=ACCENT_COLORS[i % len(ACCENT_COLORS)],
        ))
    fig.update_layout(barmode="group")
    return _apply_theme(fig, title)


# ════════════════════════════════════════════════════════════════════════════════
# ── SCATTER PLOTS ──────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def price_vs_quality_scatter(df: pd.DataFrame, sample: int = 3000) -> go.Figure:
    """
    Scatter: price vs review_score_pct.
    Colour by grouped genre, size by contrast metric.
    """
    df = df.copy()
    
    # Fix "Zero Score" Artifact: drop 0.0 scores assuming they are parsing errors/NaN fills
    df = df[df["review_score_pct"] > 0]
    
    if len(df) > sample:
        df = df.sample(sample, random_state=42)
        
    df["bubble_size"] = np.power(np.log1p(df["owners_mid"].fillna(0)), 2)
    top_genres = ["Action", "Casual", "Adventure", "Indie", "Simulation", "RPG", "Strategy"]
    df["genre_group"] = df["primary_genre"].apply(lambda g: g if g in top_genres else "Other")
    color_map = {g: ACCENT_COLORS[i % len(ACCENT_COLORS)] for i, g in enumerate(top_genres)}
    color_map["Other"] = "#64748b"

    fig = px.scatter(
        df,
        x="price",
        y="review_score_pct",
        color="genre_group",
        size="bubble_size",
        size_max=20,
        hover_data={"name": True, "price_tier": True, "price": ":.2f", "review_score_pct": ":.3f", "bubble_size": False},
        opacity=0.45,  # Lowered opacity to prevent overplotting
        labels={
            "price": "Price (USD)",
            "review_score_pct": "Review Score (ratio)",
            "genre_group": "Genre",
            "bubble_size": "Est. Ownership Size"  # Added explicit legend label
        },
        color_discrete_map=color_map,
        category_orders={"genre_group": top_genres + ["Other"]},
    )
    
    # Remove opaque bubble outlines
    fig.update_traces(marker=dict(line=dict(width=0)))
    
    # Cap X-axis at $80 to prevent extreme outlier stretching
    fig.update_layout(xaxis=dict(range=[-2, 82]))
    
    return _apply_theme(fig, "Price vs Review Quality (paid games, ≥5 reviews)")


def price_vs_ownership_scatter(df: pd.DataFrame, sample: int = 3000) -> go.Figure:
    """Scatter: price vs estimated owners (log scale) with density jitter."""
    df = df.copy()
    if len(df) > sample:
        df = df.sample(sample, random_state=42)
        
    df["log_owners"] = np.log10(df["owners_mid"].clip(lower=1))
    df["owners_jitter"] = df["log_owners"] + np.random.uniform(-0.1, 0.1, len(df))
    
    top_genres = ["Action", "Casual", "Adventure", "Indie", "Simulation", "RPG", "Strategy"]
    df["genre_group"] = df["primary_genre"].apply(lambda g: g if g in top_genres else "Other")
    color_map = {g: ACCENT_COLORS[i % len(ACCENT_COLORS)] for i, g in enumerate(top_genres)}
    color_map["Other"] = "#64748b"

    fig = px.scatter(
        df,
        x="price",
        y="owners_jitter",
        color="genre_group",
        hover_data={"name": True, "price": ":.2f", "owners_mid": ":,.0f", "owners_jitter": False},
        opacity=0.5,
        labels={
            "price": "Price (USD)",
            "owners_jitter": "Estimated Owners (log scale)",
            "genre_group": "Genre",
        },
        color_discrete_map=color_map,
        category_orders={"genre_group": top_genres + ["Other"]},
    )
    fig.update_layout(yaxis=dict(
        tickmode='array',
        tickvals=[1, 2, 3, 4, 5, 6, 7],
        ticktext=['10', '100', '1K', '10K', '100K', '1M', '10M']
    ))
    return _apply_theme(fig, "Price vs Estimated Ownership")


def quality_vs_ownership_scatter(df: pd.DataFrame, sample: int = 3000) -> go.Figure:
    """Scatter: review score vs owners_mid with density jitter."""
    df = df[df["review_score_pct"].notna()].copy()
    if len(df) > sample:
        df = df.sample(sample, random_state=42)
        
    df["log_owners"] = np.log10(df["owners_mid"].clip(lower=1))
    df["owners_jitter"] = df["log_owners"] + np.random.uniform(-0.1, 0.1, len(df))
    
    fig = px.scatter(
        df,
        x="review_score_pct",
        y="owners_jitter",
        color="price_tier",
        hover_data={"name": True, "price": ":.2f", "owners_mid": ":,.0f", "owners_jitter": False},
        opacity=0.5,
        labels={
            "review_score_pct": "Review Score (ratio)",
            "owners_jitter": "Estimated Owners (log scale)",
            "price_tier": "Price Tier",
        },
        color_discrete_sequence=ACCENT_COLORS,
    )
    fig.update_layout(yaxis=dict(
        tickmode='array',
        tickvals=[1, 2, 3, 4, 5, 6, 7],
        ticktext=['10', '100', '1K', '10K', '100K', '1M', '10M']
    ))
    return _apply_theme(fig, "Review Quality vs Estimated Ownership")


# ════════════════════════════════════════════════════════════════════════════════
# ── BOX / VIOLIN ───────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def quality_by_tier_box(df: pd.DataFrame) -> go.Figure:
    """Box plot of review_score_pct by price tier."""
    tier_order = ["Free", "Budget", "Mid-range", "Premium", "AAA"]
    df = df[df["review_score_pct"].notna()].copy()
    fig = px.box(
        df,
        x="price_tier",
        y="review_score_pct",
        category_orders={"price_tier": tier_order},
        color="price_tier",
        color_discrete_sequence=ACCENT_COLORS,
        labels={"price_tier": "Price Tier", "review_score_pct": "Review Score (ratio)"},
    )
    return _apply_theme(fig, "Review Quality Distribution by Price Tier")


def playtime_by_genre_violin(df: pd.DataFrame, max_hours: float = 100.0) -> go.Figure:
    """Violin plot of average playtime (hours) by genre."""
    df = df[df["primary_genre"].notna()].copy()
    df["avg_hrs"] = df["average_playtime_forever"] / 60
    df = df[df["avg_hrs"] <= max_hours]
    fig = px.violin(
        df,
        x="primary_genre",
        y="avg_hrs",
        color="primary_genre",
        box=True,
        color_discrete_sequence=ACCENT_COLORS,
        labels={"primary_genre": "Genre", "avg_hrs": "Avg Playtime (hours)"},
    )
    return _apply_theme(fig, "Playtime Distribution by Genre (≤100h)")


# ════════════════════════════════════════════════════════════════════════════════
# ── HEATMAPS ───────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def correlation_heatmap(corr_df: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    """Annotated correlation heatmap."""
    fig = go.Figure(go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns.tolist(),
        y=corr_df.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_df.round(2).values,
        texttemplate="%{text}",
        colorbar=dict(title="r"),
    ))
    return _apply_theme(fig, title)


def genre_metric_heatmap(pivot_df: pd.DataFrame, title: str = "", fmt: str = ".0f") -> go.Figure:
    """Heatmap for genre × tier or genre × quality-band pivots."""
    fig = go.Figure(go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns.tolist(),
        y=pivot_df.index.tolist(),
        colorscale="Viridis",
        text=pivot_df.round(0).astype(int).values,
        texttemplate="%{text}",
        colorbar=dict(title="Count"),
    ))
    return _apply_theme(fig, title)


def cohort_heatmap(df: pd.DataFrame, index_col: str, value_col: str, title: str = "") -> go.Figure:
    """Generic cohort heatmap from a pivot-ready DataFrame."""
    fig = go.Figure(go.Heatmap(
        z=df.values,
        x=df.columns.tolist(),
        y=df.index.tolist(),
        colorscale="Blues",
        colorbar=dict(title=value_col),
    ))
    return _apply_theme(fig, title)


# ════════════════════════════════════════════════════════════════════════════════
# ── TIME SERIES ────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def release_volume_line(time_df: pd.DataFrame) -> go.Figure:
    """Line chart of game releases per year."""
    fig = px.line(
        time_df,
        x="release_year",
        y="game_count",
        markers=True,
        labels={"release_year": "Release Year", "game_count": "Games Released"},
        color_discrete_sequence=[ACCENT_COLORS[0]],
    )
    fig.update_traces(line_width=2.5)
    return _apply_theme(fig, "Steam Game Releases Per Year")


def price_over_time_line(time_df: pd.DataFrame) -> go.Figure:
    """Line chart of median and mean price per year."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_df["release_year"], y=time_df["median_price"],
        name="Median Price", line=dict(color=ACCENT_COLORS[0], width=2.5), mode="lines+markers",
    ))
    fig.add_trace(go.Scatter(
        x=time_df["release_year"], y=time_df["mean_price"],
        name="Mean Price", line=dict(color=ACCENT_COLORS[1], width=2, dash="dash"), mode="lines",
    ))
    return _apply_theme(fig, "Game Pricing Over Time (paid games)")


def quality_over_time_line(time_df: pd.DataFrame) -> go.Figure:
    """Line chart of median review quality per year."""
    fig = px.line(
        time_df,
        x="release_year",
        y="median_quality",
        markers=True,
        labels={"release_year": "Release Year", "median_quality": "Median Review Score (%)"},
        color_discrete_sequence=[ACCENT_COLORS[2]],
    )
    fig.update_traces(line_width=2.5)
    return _apply_theme(fig, "Median Review Quality Over Time")


# ════════════════════════════════════════════════════════════════════════════════
# ── GAME ANALYZER CHARTS ────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def percentile_profile_bar(profile_df: pd.DataFrame, title: str = "Percentile Profile") -> go.Figure:
    """
    Horizontal bar chart showing a game's percentile rank across metrics.
    profile_df must have columns: display, market_pct (and optionally genre_pct).
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="vs Full Market",
        x=profile_df["market_pct"],
        y=profile_df["display"],
        orientation="h",
        marker_color=ACCENT_COLORS[0],
        text=profile_df["market_pct"].apply(lambda x: f"{x:.0f}th pct"),
        textposition="auto",
    ))
    if "genre_pct" in profile_df.columns:
        fig.add_trace(go.Bar(
            name="vs Genre",
            x=profile_df["genre_pct"],
            y=profile_df["display"],
            orientation="h",
            marker_color=ACCENT_COLORS[1],
            text=profile_df["genre_pct"].apply(lambda x: f"{x:.0f}th pct"),
            textposition="auto",
        ))
    fig.update_layout(barmode="group", xaxis=dict(range=[0, 100], title="Percentile Rank"))
    return _apply_theme(fig, title)


def price_context_distribution(df: pd.DataFrame, game_price: float, genre: str, genre_median: float, p25: float, p75: float) -> go.Figure:
    """Histogram of genre price distribution with target game and percentile markers."""
    df_genre = df[(df["primary_genre"] == genre) & (df["price"].notna())].copy()
    
    # Filter outliers to prevent extreme x-axis stretching
    df_genre = df_genre[df_genre["price"] <= 100]
    
    fig = px.histogram(
        df_genre,
        x="price",
        labels={"price": "Price (USD)"},
        color_discrete_sequence=[ACCENT_COLORS[0]],
        opacity=0.7,
    )
    
    fig.update_traces(xbins=dict(start=0, end=100, size=2))
    
    if not pd.isna(game_price) and game_price <= 100:
        fig.add_vline(x=game_price, line_width=3, line_color=ACCENT_COLORS[4])
        fig.add_annotation(x=game_price, y=0.95, yref='paper', text="This Game", showarrow=False, font=dict(color=ACCENT_COLORS[4], size=11), xanchor='left', xshift=5)
    
    fig.add_vline(x=genre_median, line_dash="dash", line_color="rgba(255,255,255,0.6)")
    fig.add_annotation(x=genre_median, y=0.85, yref='paper', text="Median", showarrow=False, font=dict(color="white", size=11), xanchor='right', xshift=-5)
    
    fig.add_vline(x=p25, line_dash="dot", line_color="rgba(255,255,255,0.4)")
    fig.add_vline(x=p75, line_dash="dot", line_color="rgba(255,255,255,0.4)")
    
    fig.update_layout(xaxis=dict(range=[0, 100], tickmode="linear", dtick=10))
    
    return _apply_theme(fig, f"Price Context ({genre})")


def quality_context_distribution(df: pd.DataFrame, game_quality: float, genre: str, genre_median: float) -> go.Figure:
    """Histogram of genre review score distribution with target game and median markers."""
    df_genre = df[(df["primary_genre"] == genre) & (df["review_score_pct"].notna())].copy()
    
    # Scale to percentage for visual parity
    if not df_genre.empty:
        df_genre["review_score_pct"] = df_genre["review_score_pct"] * 100
        
    fig = px.histogram(
        df_genre,
        x="review_score_pct",
        labels={"review_score_pct": "Review Score (%)"},
        color_discrete_sequence=[ACCENT_COLORS[1]],
        opacity=0.7,
    )
    
    fig.update_traces(xbins=dict(start=0, end=100, size=5))
    
    if not pd.isna(game_quality):
        fig.add_vline(x=game_quality * 100, line_width=3, line_color=ACCENT_COLORS[4])
        fig.add_annotation(x=game_quality * 100, y=0.95, yref='paper', text="This Game", showarrow=False, font=dict(color=ACCENT_COLORS[4], size=11), xanchor='left', xshift=5)
        
    fig.add_vline(x=genre_median * 100, line_dash="dash", line_color="rgba(255,255,255,0.6)")
    fig.add_annotation(x=genre_median * 100, y=0.85, yref='paper', text="Median", showarrow=False, font=dict(color="white", size=11), xanchor='right', xshift=-5)
    
    fig.update_layout(xaxis=dict(range=[0, 100], tickmode="linear", dtick=10))
    
    return _apply_theme(fig, f"Quality Context ({genre})")


def market_position_scatter(df: pd.DataFrame, game_price: float, game_quality: float, genre: str, game_name: str) -> go.Figure:
    """Scatter plot of Price vs Quality for a genre, highlighting the target game."""
    df_genre = df[(df["primary_genre"] == genre) & (df["price"].notna()) & (df["review_score_pct"].notna())].copy()
    
    if not df_genre.empty:
        df_genre["review_score_pct"] = df_genre["review_score_pct"] * 100
        df_genre["bubble_size"] = np.power(np.log1p(df_genre["owners_mid"].fillna(0)), 1.5)
        
    # Determine realistic bounds, expanding if the target game is particularly expensive
    max_x = 100
    if not pd.isna(game_price) and game_price > 100:
        max_x = game_price + 10
        
    # Filter extreme outliers just for the plot
    df_genre = df_genre[df_genre["price"] <= max_x]
        
    fig = px.scatter(
        df_genre,
        x="price",
        y="review_score_pct",
        size="bubble_size",
        size_max=15,
        opacity=0.35,
        color_discrete_sequence=["rgba(255,255,255,0.3)"],
        hover_data={"name": True, "price_tier": True, "bubble_size": False},
        labels={"price": "Price (USD)", "review_score_pct": "Review Score (%)"},
    )
    
    median_price = df_genre["price"].median()
    median_qual = df_genre["review_score_pct"].median()
    fig.add_hline(y=median_qual, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig.add_vline(x=median_price, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    
    if not pd.isna(game_price) and not pd.isna(game_quality):
        fig.add_trace(go.Scatter(
            x=[game_price],
            y=[game_quality * 100],
            mode="markers",
            marker=dict(size=16, color=ACCENT_COLORS[4], symbol="star", line=dict(width=2, color="white")),
            name=game_name,
            hovertemplate=f"{game_name}<br>Price: ${{x}}<br>Quality: {{y:.1f}}%"
        ))
        
    fig.update_layout(
        xaxis=dict(range=[0, max_x]),
        yaxis=dict(range=[0, 101])
    )
    
    return _apply_theme(fig, f"Market Position Map ({genre})")


def radar_chart(game_values: dict, benchmark_values: dict, labels: list[str]) -> go.Figure:
    """
    Radar chart comparing a game against a benchmark (e.g., genre median).
    Values should be pre-normalised to a common scale (e.g., percentiles 0–100).
    """
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[game_values.get(l, 0) for l in labels] + [game_values.get(labels[0], 0)],
        theta=labels + [labels[0]],
        fill="toself",
        name="This Game",
        line_color=ACCENT_COLORS[0],
        fillcolor="rgba(124,58,237,0.25)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[benchmark_values.get(l, 0) for l in labels] + [benchmark_values.get(labels[0], 0)],
        theta=labels + [labels[0]],
        fill="toself",
        name="Genre Median",
        line_color=ACCENT_COLORS[1],
        fillcolor="rgba(59,130,246,0.20)",
    ))
    fig.update_layout(polar=dict(
        radialaxis=dict(visible=True, range=[0, 100]),
        bgcolor=PLOTLY_BG_COLOR,
    ))
    return _apply_theme(fig, "Game Profile vs Genre Median")


def similar_games_heatmap(sim_df: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """
    Heatmap comparing similar games across key metrics.
    Rows = games, Columns = metrics.
    Values are normalised within each column.
    """
    sub = sim_df[["name"] + [m for m in metrics if m in sim_df.columns]].copy()
    sub = sub.set_index("name")

    # Normalise per-column to 0–1
    norm = (sub - sub.min()) / (sub.max() - sub.min() + 1e-9)

    fig = go.Figure(go.Heatmap(
        z=norm.values,
        x=norm.columns.tolist(),
        y=norm.index.tolist(),
        colorscale=[[0.0, "rgba(0,0,0,0)"], [1.0, ACCENT_COLORS[1]]], # Custom cyberpunk scale
        text=sub.round(2).values,
        texttemplate="%{text}",
    ))
    return _apply_theme(fig, "Similar Games Comparison")


def benchmark_marker_chart(profile_df: pd.DataFrame, game_name: str = "Game") -> go.Figure:
    """
    Dot-plot showing each metric's value relative to market median, anchored with lines.
    profile_df: output of benchmarks.percentile_profile()
    """
    fig = go.Figure()
    
    # Draw a line from 50 (median) to the point to ground the chart
    for _, row in profile_df.iterrows():
        pct = row["market_pct"]
        color = ACCENT_COLORS[1] if pct >= 50 else ACCENT_COLORS[0]
        
        fig.add_trace(go.Scatter(
            x=[50, pct],
            y=[row["display"], row["display"]],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo="skip"
        ))
        
        fig.add_trace(go.Scatter(
            x=[pct],
            y=[row["display"]],
            mode="markers",
            marker=dict(size=14, color=color, line=dict(color="white", width=1)),
            name=row["display"],
            showlegend=False,
            hovertemplate=f"{row['display']}: {row['value']:.2f}<br>Market Pct: {pct:.0f}th",
        ))
        
    fig.add_vline(x=50, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                  annotation_text="Median (50th)")
    fig.update_layout(
        xaxis=dict(range=[0, 100], title="Percentile Rank (0=lowest, 100=highest)"),
        yaxis=dict(title="")
    )
    return _apply_theme(fig, f"{game_name} — Market Position Markers")


def market_gap_signal_bar(gap_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Bar chart of Market Gap Signal by genre × tier (top N segments)."""
    df = gap_df.head(top_n).copy()
    df["segment"] = df["primary_genre"] + " / " + df["price_tier"]
    fig = px.bar(
        df,
        x="gap_signal_norm",
        y="segment",
        orientation="h",
        color="gap_signal_norm",
        color_continuous_scale=["#3B82F6", "#7C3AED"],
        labels={"gap_signal_norm": "Market Gap Signal (0–100)", "segment": "Genre / Tier"},
        hover_data=["game_count", "median_owners", "median_quality"],
    )
    fig.update_coloraxes(showscale=False)
    return _apply_theme(fig, "Market Gap Signal — Top Segments by Analytical Indicator")


def cluster_scatter(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    """Scatter plot coloured by cluster label."""
    sample_df = df.sample(min(5000, len(df)), random_state=42)
    fig = px.scatter(
        sample_df,
        x=x, y=y,
        color="cluster_label",
        hover_data=["name", "price", "review_score_pct"],
        opacity=0.65,
        color_discrete_sequence=ACCENT_COLORS,
        labels={"cluster_label": "Segment"},
    )
    return _apply_theme(fig, f"Market Segments: {x} vs {y}")


def parallel_coordinates_chart(df: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """Parallel coordinates for multidimensional market comparison."""
    available = [m for m in metrics if m in df.columns]
    sample_df = df[available + ["cluster_id"]].dropna().sample(
        min(2000, len(df)), random_state=42
    )
    fig = px.parallel_coordinates(
        sample_df,
        color="cluster_id",
        dimensions=available,
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={m: m.replace("_", " ").title() for m in available},
    )
    return _apply_theme(fig, "Parallel Coordinates — Market Segments")
