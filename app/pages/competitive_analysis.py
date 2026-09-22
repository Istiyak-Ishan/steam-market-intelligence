"""
Competitive Analysis page — Wrapper for Game Matchup and Find Similar Games.
"""
import streamlit as st
import pandas as pd

from app.pages import game_comparison, similar_games

def render(df: pd.DataFrame, models=None) -> None:
    st.markdown('<div class="hero-header"><div class="hero-title">⚔️ Competitive Analysis</div><div class="hero-subtitle">Evaluate games against each other or find nearest neighbors.</div></div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Game Matchup", "Find Similar Games"])
    
    with tab1:
        game_comparison.render(df, models, hide_header=True)
        
    with tab2:
        similar_games.render(df, hide_header=True)
