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







# ════════════════════════════════════════════════════════════════════════════════
# ── BAR CHARTS ─────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════







# ════════════════════════════════════════════════════════════════════════════════
# ── SCATTER PLOTS ──────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════







# ════════════════════════════════════════════════════════════════════════════════
# ── BOX / VIOLIN ───────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════





# ════════════════════════════════════════════════════════════════════════════════
# ── HEATMAPS ───────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════







# ════════════════════════════════════════════════════════════════════════════════
# ── TIME SERIES ────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════







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








