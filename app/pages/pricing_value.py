"""
Pricing & Value page — Price distributions, value mapping, and cohort trends.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.config import ACCENT_COLORS, PLOTLY_BG_COLOR, PLOTLY_PAPER_BG, PLOTLY_FONT_COLOR
from src.benchmarks import percentile_profile

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

def render(df: pd.DataFrame) -> None:
    st.markdown('<div class="hero-header"><div class="hero-title">💰 Pricing & Value</div><div class="hero-subtitle">Pricing distributions, commercial value benchmarking, and historical pricing trends.</div></div>', unsafe_allow_html=True)
    
    # Pre-process
    paid = df[(df["price"] > 0) & (df["price"] <= 100)].copy()
    
    # ── Row 1: Distributions ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Price Distributions</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    
    # 1. Price Histogram
    with c1:
        fig1 = px.histogram(paid, x="price", nbins=50, color_discrete_sequence=[ACCENT_COLORS[0]])
        fig1.update_layout(**_L(title="Price Histogram (paid ≤ $100)", margin=dict(t=40, r=10, b=40, l=40)))
        st.plotly_chart(fig1, use_container_width=True)
        
    # 2. Genre Boxplot
    with c2:
        top_genres = paid["primary_genre"].value_counts().head(10).index
        box_df = paid[paid["primary_genre"].isin(top_genres)]
        fig2 = px.box(box_df, x="price", y="primary_genre", color="primary_genre", orientation="h")
        fig2.update_layout(**_L(title="Price by Genre (Top 10)", margin=dict(t=40, r=10, b=40, l=80)), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        
    # 3. Price-Tier Distribution
    with c3:
        tier_counts = df["price_tier"].value_counts().reset_index()
        fig3 = px.pie(tier_counts, values="count", names="price_tier", hole=0.4, 
                      color_discrete_sequence=ACCENT_COLORS)
        fig3.update_layout(**_L(title="Price-Tier Distribution", margin=dict(t=40, r=10, b=40, l=40)))
        st.plotly_chart(fig3, use_container_width=True)
        
    # ── Row 2: Scatters ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Pricing Economics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        samp1 = paid.dropna(subset=["review_score_pct"]).sample(min(3000, len(paid)), random_state=42)
        fig4 = px.scatter(samp1, x="price", y="review_score_pct", opacity=0.3, color_discrete_sequence=[ACCENT_COLORS[1]])
        fig4.update_layout(**_L(title="Price × Quality (Sampled)"))
        st.plotly_chart(fig4, use_container_width=True)
    with c2:
        samp2 = paid.dropna(subset=["owners_mid"]).sample(min(3000, len(paid)), random_state=42)
        fig5 = px.scatter(samp2, x="price", y="owners_mid", opacity=0.3, log_y=True, color_discrete_sequence=[ACCENT_COLORS[2]])
        fig5.update_layout(**_L(title="Price × Ownership (Sampled)"))
        st.plotly_chart(fig5, use_container_width=True)

    # ── Row 3: Value Score ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Value Analysis (Quality / Price)</div>', unsafe_allow_html=True)
    paid_val = paid[paid["review_score_pct"].notna() & (paid["price"] >= 1.99)].copy()
    paid_val["value_score"] = (paid_val["review_score_pct"] * 100) / np.log2(paid_val["price"] + 1)
    
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        fig6 = px.histogram(paid_val, x="value_score", nbins=50, color_discrete_sequence=[ACCENT_COLORS[3]])
        fig6.update_layout(**_L(title="Value-Score Distribution (≥ $1.99)"))
        st.plotly_chart(fig6, use_container_width=True)
        
    with c2:
        g_val = paid_val.groupby("primary_genre")["value_score"].median().sort_values().tail(10).reset_index()
        fig7 = px.bar(g_val, x="value_score", y="primary_genre", orientation="h", color_discrete_sequence=[ACCENT_COLORS[4]])
        fig7.update_layout(**_L(title="Median Value Score by Genre", margin=dict(t=40, r=10, b=40, l=100)))
        st.plotly_chart(fig7, use_container_width=True)
        
    with c3:
        # Price x Value Heatmap
        fig8 = px.density_heatmap(paid_val, x="price", y="value_score", nbinsx=20, nbinsy=20, color_continuous_scale="Viridis")
        fig8.update_layout(**_L(title="Price × Value Density"))
        st.plotly_chart(fig8, use_container_width=True)

    # ── Row 4: Trends & Benchmarks ────────────────────────────────────────────
    st.markdown('<div class="section-header">Trends & Benchmarks</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        # Cohort Trend
        cohort = df[df["release_year"].between(2010, 2025)].groupby("release_year")["price"].median().reset_index()
        fig9 = px.line(cohort, x="release_year", y="price", markers=True, color_discrete_sequence=[ACCENT_COLORS[0]])
        fig9.update_layout(**_L(title="Median Price Cohort Trend (2010-2025)"))
        st.plotly_chart(fig9, use_container_width=True)
        
    with c2:
        # Genre Pricing Benchmark
        genres = df["primary_genre"].value_counts().head(8).index
        bench_data = []
        for g in genres:
            gdf = df[(df["primary_genre"] == g) & (df["price"] > 0)]
            if not gdf.empty:
                bench_data.append({
                    "Genre": g,
                    "25th": gdf["price"].quantile(0.25),
                    "Median": gdf["price"].median(),
                    "75th": gdf["price"].quantile(0.75)
                })
        bdf = pd.DataFrame(bench_data)
        fig10 = go.Figure()
        fig10.add_trace(go.Bar(name='25th Pct', x=bdf['Genre'], y=bdf['25th'], marker_color=ACCENT_COLORS[0]))
        fig10.add_trace(go.Bar(name='Median', x=bdf['Genre'], y=bdf['Median'], marker_color=ACCENT_COLORS[1]))
        fig10.add_trace(go.Bar(name='75th Pct', x=bdf['Genre'], y=bdf['75th'], marker_color=ACCENT_COLORS[2]))
        fig10.update_layout(barmode='group', **_L(title="Genre Pricing Benchmark (25th, 50th, 75th)"))
        st.plotly_chart(fig10, use_container_width=True)
