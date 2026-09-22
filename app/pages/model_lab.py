"""
model_lab.py — Temporarily disabled for Model Upgrades.
"""
import streamlit as st

def render(df=None, models=None, **kwargs) -> None:
    st.markdown('<div class="hero-header"><div class="hero-title">🧪 Model Inspector</div><div class="hero-subtitle">Model diagnostics and feature importance.</div></div>', unsafe_allow_html=True)
    st.warning("⚠️ **Model Lab is currently disabled.** We have upgraded the backend to use high-precision HistGradientBoostingRegressors. The model inspector is currently being rewritten to support explaining these complex gradient-boosted ensembles. Please check back later.")