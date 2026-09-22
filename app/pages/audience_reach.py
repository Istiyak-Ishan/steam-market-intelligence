"""
Audience & Reach page — Wrapper for Player Discovery and Platform & Global.
"""
import streamlit as st
import pandas as pd

from app.pages import player_discovery, platform_global

def render(df: pd.DataFrame, models=None) -> None:
    st.markdown('<div class="hero-header"><div class="hero-title">🌍 Audience & Reach</div><div class="hero-subtitle">Player discovery, language, and platform demographics.</div></div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Player Discovery", "Platform & Global"])
    
    with tab1:
        player_discovery.render(df, hide_header=True)
        
    with tab2:
        platform_global.render(df, hide_header=True)
