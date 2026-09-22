"""
Market Map page — Interactive multidimensional scatter analysis.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.config import ACCENT_COLORS, PLOTLY_BG_COLOR, PLOTLY_PAPER_BG, PLOTLY_FONT_COLOR

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

def _get_scatter(df: pd.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str, title: str, highlight_game: str = "", log_x: bool = False, log_y: bool = False) -> go.Figure:
    sub = df[df[x_col].notna() & df[y_col].notna() & df["owners_mid"].notna() & (df["price"] > 0)].copy()
    if sub.empty: return go.Figure()
    
    # Cap size and calculate sizes for ownership
    sub["bubble_size"] = np.power(np.log10(sub["owners_mid"].clip(lower=1)), 1.8)
    
    # Highlight logic
    if highlight_game and highlight_game in sub["name"].values:
        sub["opacity"] = np.where(sub["name"] == highlight_game, 1.0, 0.04)
        sub = sub.sort_values("opacity") # bring highlighted to front
        sub["marker_line_width"] = np.where(sub["name"] == highlight_game, 2.0, 0.0)
    else:
        sub["opacity"] = 0.35
        sub["marker_line_width"] = 0.5
        
    top_genres = ["Action", "Casual", "Adventure", "Indie", "Simulation", "RPG", "Strategy"]
    sub["color_genre"] = sub["primary_genre"].apply(lambda g: g if g in top_genres else "Other")
    
    fig = px.scatter(
        sub, x=x_col, y=y_col, size="bubble_size", color="color_genre",
        hover_name="name", hover_data={x_col: True, y_col: True, "owners_mid": ":,.0f", "bubble_size": False, "opacity": False, "marker_line_width": False},
        opacity=sub["opacity"],
        size_max=22,
        color_discrete_sequence=[ACCENT_COLORS[0], ACCENT_COLORS[1], ACCENT_COLORS[2], ACCENT_COLORS[3], "#f43f5e", "#06b6d4", "#ec4899", "#64748b"],
        labels={x_col: x_label, y_col: y_label, "owners_mid": "Est. Owners"}
    )
    
    x_cfg = dict(title=x_label, showgrid=True, gridcolor="#1e2030")
    if log_x: x_cfg["type"] = "log"
    y_cfg = dict(title=y_label, showgrid=True, gridcolor="#1e2030")
    if log_y: y_cfg["type"] = "log"
    
    fig.update_layout(
        **_L(title=title), 
        xaxis=x_cfg, 
        yaxis=y_cfg, 
        legend=dict(orientation="h", y=1.08, x=0, title="Genre")
    )
    
    # Apply dynamic stroke width based on highlight
    fig.update_traces(marker=dict(line=dict(color="white")))
    for trace in fig.data:
        # Plotly Express groups by color, so we need to set the line width array for each group
        mask = sub["color_genre"] == trace.name
        if not mask.any(): continue
        trace.marker.line.width = sub.loc[mask, "marker_line_width"].values

    return fig

def render(df: pd.DataFrame, hide_header: bool = False) -> None:
    if not hide_header:
        st.markdown('<div class="hero-header"><div class="hero-title">🗺️ Market Map</div><div class="hero-subtitle">Interactive multidimensional scatter analysis across the commercial landscape. (Size = Ownership, Color = Genre)</div></div>', unsafe_allow_html=True)
    
    with st.expander("🔍 Filters & Highlighting", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            genres = df["primary_genre"].dropna().unique().tolist()
            sel_genres = st.multiselect("Filter by Genre", sorted(genres))
        with c2:
            max_p = float(df["price"].max())
            price_range = st.slider("Price Range", 0.0, min(100.0, max_p), (0.0, min(70.0, max_p)), 5.0)
        with c3:
            highlight = st.selectbox("Highlight specific game (fades others)", [""] + df["name"].dropna().tolist(), index=0)
            
    fdf = df.copy()
    if sel_genres:
        fdf = fdf[fdf["primary_genre"].isin(sel_genres)]
    fdf = fdf[fdf["price"].between(price_range[0], price_range[1])]
    fdf["review_pct"] = fdf["review_score_pct"] * 100
    
    st.plotly_chart(_get_scatter(fdf, "price", "review_pct", "Price ($)", "Review Score (%)", "Price × Review Quality", highlight), use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_get_scatter(fdf, "review_pct", "owners_mid", "Review Score (%)", "Est. Owners (Log)", "Quality × Ownership", highlight, log_y=True), use_container_width=True)
    with c2:
        st.plotly_chart(_get_scatter(fdf, "price", "owners_mid", "Price ($)", "Est. Owners (Log)", "Price × Ownership", highlight, log_y=True), use_container_width=True)
        
    st.plotly_chart(_get_scatter(fdf, "peak_ccu", "owners_mid", "Peak CCU (Log)", "Est. Owners (Log)", "CCU × Ownership", highlight, log_x=True, log_y=True), use_container_width=True)
