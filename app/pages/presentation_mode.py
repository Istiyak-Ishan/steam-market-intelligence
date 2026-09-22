"""
presentation_mode.py -- Guided 10-step sequential presentation of the project.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import src.visualization as vz

from src.config import ACCENT_COLORS

def render(df: pd.DataFrame, models: dict) -> None:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">📽️ Presentation Mode</div>
        <div class="hero-subtitle">
            Executive Showcase: Steam Market Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "presentation_step" not in st.session_state:
        st.session_state["presentation_step"] = 1

    total_steps = 10
    step = st.session_state["presentation_step"]

    # Navigation Controls
    st.markdown("<br>", unsafe_allow_html=True)
    c_prev, c_prog, c_next = st.columns([1, 2, 1])
    
    with c_prev:
        if step > 1:
            if st.button("⬅️ PREVIOUS", use_container_width=True):
                st.session_state["presentation_step"] -= 1
                st.rerun()
    with c_prog:
        st.progress(step / total_steps)
        st.markdown(f"<div style='text-align: center; color: var(--text-muted); font-family: JetBrains Mono; font-size: 0.9rem; margin-top: 5px;'>STEP {step} OF {total_steps}</div>", unsafe_allow_html=True)
    with c_next:
        if step < total_steps:
            if st.button("NEXT ➡️", use_container_width=True):
                st.session_state["presentation_step"] += 1
                st.rerun()

    st.markdown("<br><hr style='border-color: var(--border-default);'>", unsafe_allow_html=True)

    # Step Content
    if step == 1:
        st.markdown('<div class="section-header">1. THE BUSINESS PROBLEM</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="info-box" style="border-color: #EF4444; background: rgba(239,68,68,0.05);">
            <h3 style="color: #EF4444; font-family: Orbitron;">THE BLIND SPOT</h3>
            Indie developers and AA publishers often enter the Steam marketplace based on instinct rather than data.<br><br>
            • <b>Pricing</b> is based on gut feeling.<br>
            • <b>Feature scope</b> is bloated without market validation.<br>
            • <b>Competitive positioning</b> is complete guesswork.
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="info-box" style="border-color: #10B981; background: rgba(16,185,129,0.05);">
            <h3 style="color: #10B981; font-family: Orbitron;">THE SOLUTION</h3>
            An empirically driven, machine-learning-backed analytical tool that definitively answers:<br><br>
            • What is the natural price tier for this game's scope?<br>
            • Is there a market gap for this genre?<br>
            • How does this game stack up against direct historical competitors?
            </div>
            """, unsafe_allow_html=True)

    elif step == 2:
        st.markdown('<div class="section-header">2. DATA FOUNDATION & SCALE</div>', unsafe_allow_html=True)
        st.markdown("We are operating on a massive historical dataset spanning decades of Steam releases.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-label">Analyzed Titles</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">1997 - 2024</div>
                <div class="metric-label">Date Range</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">Random Forest</div>
                <div class="metric-label">ML Architecture</div>
            </div>
            """, unsafe_allow_html=True)
            
        top_genres = df["primary_genre"].value_counts().head(5)
        fig = px.bar(top_genres, orientation='h', color_discrete_sequence=[ACCENT_COLORS[0]])
        fig = vz._apply_theme(fig, "Top 5 Genres by Volume")
        st.plotly_chart(fig, use_container_width=True)

    elif step == 3:
        st.markdown('<div class="section-header">3. THE MARKET MAP: Price vs Quality</div>', unsafe_allow_html=True)
        st.markdown("At a macro level, we observe a distinct clustering of games where price naturally limits quality expectations, but extreme outliers exist. Our platform maps this landscape.")
        sample = df.sample(min(3000, len(df)))
        fig = vz.price_vs_quality_scatter(sample)
        st.plotly_chart(fig, use_container_width=True)

    elif step == 4:
        st.markdown('<div class="section-header">4. MICRO-ANALYSIS: The Game Analyzer</div>', unsafe_allow_html=True)
        st.markdown("Macro trends are interesting, but publishers need to drill into specific titles. The **Game Analyzer** (available in the sidebar) allows users to search any game or build a hypothetical one.")
        st.info("💡 **Capabilities:** Real-time percentile generation, Radar Profile comparison against genre medians, and Anomaly Detection.")
        
        st.markdown("""
        <div class="info-box">
        By examining a game microscopically, we can definitively say whether a game is overperforming its price tier or severely lacking in a specific feature (like supported languages) compared to its peers.
        </div>
        """, unsafe_allow_html=True)
        
        # Add visual example of radar chart
        st.markdown("<br>", unsafe_allow_html=True)
        labels = ["Price", "Review Score", "Owners", "Peak CCU", "Playtime"]
        game_vals = {"Price": 75, "Review Score": 90, "Owners": 60, "Peak CCU": 55, "Playtime": 80}
        bench_vals = {"Price": 50, "Review Score": 50, "Owners": 50, "Peak CCU": 50, "Playtime": 50}
        fig = vz.radar_chart(game_vals, bench_vals, labels)
        st.plotly_chart(fig, use_container_width=True)

    elif step == 5:
        st.markdown('<div class="section-header">5. COMPETITIVE BENCHMARKING</div>', unsafe_allow_html=True)
        st.markdown("How does an average $15 Indie Action game compare to the rest of the market? We visually benchmark metrics.")
        
        dummy_data = pd.DataFrame({
            "display": ["Price ($)", "Review Score (%)", "Est. Owners"],
            "market_pct": [65, 82, 45],
            "value": [14.99, 0.85, 25000]
        })
        fig = vz.benchmark_marker_chart(dummy_data, "Indie Action (Example Profile)")
        st.plotly_chart(fig, use_container_width=True)

    elif step == 6:
        st.markdown('<div class="section-header">6. THE SIMILARITY ENGINE</div>', unsafe_allow_html=True)
        st.markdown("Publishers constantly ask: *'Who are we really competing with?'*")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            Instead of guessing, the platform uses a **9-feature standardized Cosine Similarity Engine**.<br><br>
            It mathematically scans the entire dataset to find the closest empirical competitors based on mechanics, scope, age, and pricing.
            """)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">Cosine</div>
                <div class="metric-label">Distance Metric</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.success("This allows publishers to set realistic benchmarks based on games that actually look and feel like theirs, rather than broad genre averages.")

        st.markdown("<br>", unsafe_allow_html=True)
        sim_df = pd.DataFrame({
            "name": ["Your Game", "Competitor A", "Competitor B", "Competitor C"],
            "price": [19.99, 14.99, 19.99, 24.99],
            "review_score": [0.85, 0.88, 0.75, 0.90],
            "owners": [50000, 45000, 60000, 30000]
        })
        fig = vz.similar_games_heatmap(sim_df, ["price", "review_score", "owners"])
        st.plotly_chart(fig, use_container_width=True)

    elif step == 7:
        st.markdown('<div class="section-header">7. MARKET SEGMENTATION (K-Means)</div>', unsafe_allow_html=True)
        st.markdown("Genres (Action, RPG) are subjective and often unhelpful for business decisions. We deployed **K-Means clustering (k=5)** on numeric features to discover hidden market archetypes.")
        
        st.markdown("""
        **Discovered Archetypes include:**
        - 🏆 **Premium Hits:** High price, massive engagement, AAA scope.
        - 🧩 **Budget Casual:** Very cheap, highly saturated, low average playtime.
        - 🎯 **Mid-Range Niche:** Specialized games with deeply engaged audiences.
        
        *(Users can explore these clusters deeply in the **Segmentation** tab).*
        """)

    elif step == 8:
        st.markdown('<div class="section-header">8. THE CROWN JEWEL: Publisher Scenario Engine</div>', unsafe_allow_html=True)
        st.markdown("The **Publisher Studio** provides a dual-scenario What-If engine.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="info-box" style="border-color: #3B82F6;">
            <h4 style="color: #3B82F6; margin:0;">Base Scenario</h4>
            Input your current plan.<br>
            <i>Example: $19.99, 5 Languages, Single-Player Action.</i>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="info-box" style="border-color: #F59E0B;">
            <h4 style="color: #F59E0B; margin:0;">What-If Scenario</h4>
            Tweak the variables.<br>
            <i>Example: Cut price to $14.99, add 10 Languages.</i>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>The Machine Learning models instantly predict the shifts in **Price Tier** and **Value Score**, allowing publishers to optimize their launch strategy mathematically.", unsafe_allow_html=True)

    elif step == 9:
        st.markdown('<div class="section-header">9. KEY STRATEGIC INSIGHTS</div>', unsafe_allow_html=True)
        st.markdown("Through building this platform, the data revealed several critical insights:")
        
        st.markdown("""
        1. 🌍 **Language ROI has a ceiling:** There are diminishing returns after ~8 supported languages for Indie titles. Adding 20 languages does not linearly scale owners.
        2. 💰 **Price Elasticity by Genre:** 'Strategy' and 'Simulation' gamers tolerate $20+ price points far better than 'Casual' gamers, who expect budget pricing.
        3. 📉 **The 70% Quality Floor:** Titles falling below a 70% review score face near-zero organic discovery and algorithmic suppression by Steam, regardless of how cheap they are.
        """)

    elif step == 10:
        st.markdown('<div class="section-header">10. METHODOLOGY & CAVEATS</div>', unsafe_allow_html=True)
        st.markdown("Transparency is critical for a data tool. Users must understand the bounds of the machine.")
        
        st.warning("""
        **CRITICAL LIMITATIONS:**
        - **Correlation ≠ Causation:** All predictions and metrics are based on historical observational associations. The ML models do not guarantee commercial success.
        - **Data Provenance:** Free-to-play games were explicitly excluded from the pricing models to avoid extreme skew.
        - **Estimates:** Owner counts and revenue approximations are based on bucket midpoints and review-multipliers.
        """)
        
        st.markdown("""
        <div class="hero-header" style="margin-top: 20px; text-align: center;">
            <div class="hero-title" style="font-size: 1.8rem; color: #10B981;">✅ END OF PRESENTATION</div>
            <div class="hero-subtitle" style="margin-top: 10px;">
                You may now explore the live application using the sidebar navigation.
            </div>
        </div>
        """, unsafe_allow_html=True)
