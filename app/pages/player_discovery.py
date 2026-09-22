"""
Player & Discovery page — Reach, engagement, and cross-metric analysis.
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

def render(df: pd.DataFrame) -> None:
    st.markdown('<div class="hero-header"><div class="hero-title">👥 Player & Discovery</div><div class="hero-subtitle">Engagement, reach, and community discovery cross-metrics.</div></div>', unsafe_allow_html=True)
    
    # ── Row 1: Base Metrics Dashboard ─────────────────────────────────────────
    st.markdown('<div class="section-header">Global Base Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total Est. Owners", f"{df['owners_mid'].sum():,.0f}")
    with c2: st.metric("Total Reviews", f"{df['total_review'].sum():,.0f}")
    with c3: st.metric("Total Recommendations", f"{df['recommendations'].sum():,.0f}")
    with c4: st.metric("Peak CCU (All Time)", f"{df['peak_ccu'].max():,.0f}")
    with c5: st.metric("Median Playtime", f"{df[df['average_playtime_forever']>0]['average_playtime_forever'].median():,.0f} min")
    
    st.markdown("""
    <div class="info-box">
        <strong>⚠️ Correlation vs Causation:</strong> 
        The trendlines plotted below show statistical associations (correlation), not causal effects. 
        For example, high review quality is associated with higher ownership, but we cannot mathematically 
        prove quality inherently causes higher sales independent of marketing, IP strength, and platform visibility.
    </div>
    """, unsafe_allow_html=True)

    # Pre-process for cross-metrics
    fdf = df[df["owners_mid"].notna() & (df["owners_mid"] > 1000)].copy()
    
    # ── Row 2: Cross-Metrics (Quality) ────────────────────────────────────────
    st.markdown('<div class="section-header">Quality associations</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    # Quality vs Reach (Reviews)
    with c1:
        samp = fdf.dropna(subset=["review_score_pct", "total_review"]).sample(min(3000, len(fdf)), random_state=42)
        fig1 = px.scatter(samp, x="review_score_pct", y="total_review", log_y=True, opacity=0.3,
                          trendline="ols" if len(samp) > 30 else None,
                          color_discrete_sequence=[ACCENT_COLORS[0]])
        fig1.update_layout(**_L(title="Quality vs Reach (Reviews)"))
        st.plotly_chart(fig1, use_container_width=True)

    # Quality vs Ownership
    with c2:
        samp = fdf.dropna(subset=["review_score_pct", "owners_mid"]).sample(min(3000, len(fdf)), random_state=42)
        fig2 = px.scatter(samp, x="review_score_pct", y="owners_mid", log_y=True, opacity=0.3,
                          trendline="ols" if len(samp) > 30 else None,
                          color_discrete_sequence=[ACCENT_COLORS[1]])
        fig2.update_layout(**_L(title="Quality vs Estimated Ownership"))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 3: Cross-Metrics (Engagement) ─────────────────────────────────────
    st.markdown('<div class="section-header">Engagement associations</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    # Recommendations vs Ownership
    with c1:
        samp = fdf[fdf["recommendations"] > 0].dropna(subset=["recommendations", "owners_mid"]).sample(min(3000, len(fdf)), random_state=42)
        fig3 = px.scatter(samp, x="recommendations", y="owners_mid", log_x=True, log_y=True, opacity=0.3,
                          trendline="ols" if len(samp) > 30 else None,
                          color_discrete_sequence=[ACCENT_COLORS[2]])
        fig3.update_layout(**_L(title="Recommendations vs Ownership"))
        st.plotly_chart(fig3, use_container_width=True)
        
    # CCU vs Ownership
    with c2:
        samp = fdf[fdf["peak_ccu"] > 0].dropna(subset=["peak_ccu", "owners_mid"]).sample(min(3000, len(fdf)), random_state=42)
        fig4 = px.scatter(samp, x="peak_ccu", y="owners_mid", log_x=True, log_y=True, opacity=0.3,
                          trendline="ols" if len(samp) > 30 else None,
                          color_discrete_sequence=[ACCENT_COLORS[3]])
        fig4.update_layout(**_L(title="Peak CCU vs Ownership"))
        st.plotly_chart(fig4, use_container_width=True)
