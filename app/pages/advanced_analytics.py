"""
Advanced Analytics page — Wrapper for Model Lab, Segmentation, and Anomaly Finder.
"""
import streamlit as st
import pandas as pd

from app.pages import model_lab, segmentation_page, anomaly_finder

def render(df: pd.DataFrame, models=None) -> None:
    st.markdown('<div class="hero-header"><div class="hero-title">🧪 Advanced Analytics</div><div class="hero-subtitle">Machine learning insights: Predictions, Clustering, and Outliers.</div></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Predictive Lab", "Clustering", "Outlier Detection"])
    
    with tab1:
        model_lab.render(df, models, hide_header=True)
        
    with tab2:
        segmentation_page.render(df, hide_header=True)
        
    with tab3:
        anomaly_finder.render(df, hide_header=True)
