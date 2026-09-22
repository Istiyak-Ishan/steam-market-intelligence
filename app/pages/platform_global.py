"""
Platform & Global page — Platform coverage, OS support, and localization analytics.
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

def render(df: pd.DataFrame, hide_header: bool = False) -> None:
    if not hide_header:
        st.markdown('<div class="hero-header"><div class="hero-title">🌍 Platform & Global</div><div class="hero-subtitle">Operating system coverage and localization analytics.</div></div>', unsafe_allow_html=True)
    
    # Pre-process platform combinations
    df_plat = df.copy()
    
    def get_combo(r):
        combo = []
        if r.get("windows"): combo.append("Win")
        if r.get("mac"): combo.append("Mac")
        if r.get("linux"): combo.append("Lin")
        return "+".join(combo) if combo else "Unknown"
        
    df_plat["platform_combo"] = df_plat.apply(get_combo, axis=1)
    
    # ── Row 1: Platform Reach ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Platform Support</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    with c1:
        # Platform combinations distribution
        combo_counts = df_plat["platform_combo"].value_counts().reset_index()
        fig1 = px.bar(combo_counts, x="platform_combo", y="count", color_discrete_sequence=[ACCENT_COLORS[0]])
        fig1.update_layout(**_L(title="Platform Combinations (Game Count)"))
        st.plotly_chart(fig1, use_container_width=True)
        
    with c2:
        # Ownership by Platform Combo
        own_combo = df_plat.groupby("platform_combo")["owners_mid"].median().reset_index()
        fig2 = px.bar(own_combo, x="platform_combo", y="owners_mid", color_discrete_sequence=[ACCENT_COLORS[1]])
        fig2.update_layout(**_L(title="Median Ownership by Platform Combo"))
        st.plotly_chart(fig2, use_container_width=True)
        
    with c3:
        # Review score by Platform Combo
        rev_combo = df_plat.dropna(subset=["review_score_pct"])
        fig3 = px.box(rev_combo, x="platform_combo", y="review_score_pct", color="platform_combo")
        fig3.update_layout(**_L(title="Review Score by Platform Combo"), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Row 2: Global Reach (Languages) ───────────────────────────────────────
    st.markdown('<div class="section-header">Localization (Supported Languages)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    df_lang = df[df["languages_count"].notna()].copy()
    
    with c1:
        # Language Count Distribution
        fig4 = px.histogram(df_lang, x="languages_count", nbins=30, color_discrete_sequence=[ACCENT_COLORS[2]])
        fig4.update_layout(**_L(title="Language Count Distribution"))
        st.plotly_chart(fig4, use_container_width=True)
        
    with c2:
        # Language Count vs Ownership
        samp = df_lang.dropna(subset=["owners_mid"]).sample(min(3000, len(df_lang)), random_state=42)
        fig5 = px.box(samp, x="languages_count", y="owners_mid", log_y=True, color_discrete_sequence=[ACCENT_COLORS[3]])
        fig5.update_layout(**_L(title="Language Count vs Ownership"))
        st.plotly_chart(fig5, use_container_width=True)
        
    # Localization by Genre Heatmap
    st.markdown('<div class="section-header">Localization by Genre</div>', unsafe_allow_html=True)
    g_lang = df_lang.groupby("primary_genre")["languages_count"].median().sort_values(ascending=False).head(15).reset_index()
    fig6 = px.bar(g_lang, x="languages_count", y="primary_genre", orientation="h", color="languages_count", color_continuous_scale="Blues")
    fig6.update_layout(**_L(title="Median Supported Languages by Genre (Top 15)"))
    st.plotly_chart(fig6, use_container_width=True)
