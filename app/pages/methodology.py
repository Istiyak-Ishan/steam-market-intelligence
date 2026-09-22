"""
Methodology page - Displays the embedded Jupyter EDA Notebook.
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from src.config import PROJECT_ROOT

def render(df=None, models=None) -> None:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">📖 Methodology & EDA Notebook</div>
        <div class="hero-subtitle">
            Direct interactive export of the foundational Exploratory Data Analysis (EDA) process. 
            Each section includes auto-generated key findings derived from the raw data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    html_path = PROJECT_ROOT / "notebooks" / "Strategic_EDA_output.html"

    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            html_data = f.read()
        
        # We wrap the HTML in a container to give it some padding and height
        with st.container(border=True):
            components.html(html_data, height=1200, scrolling=True)
    else:
        st.error("⚠️ EDA_output.html not found. Please run the notebook export script first.")
