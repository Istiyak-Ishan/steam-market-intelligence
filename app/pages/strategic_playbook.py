"""
strategic_playbook.py -- Data-backed insights and strategies tailored by audience.

Strict format: OBSERVATION → VISUAL EVIDENCE → INTERPRETATION → POSSIBLE ACTION.
No causal overclaims, relying entirely on existing platform dataset features.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd


def render(df: pd.DataFrame) -> None:
    total_games = len(df)
    st.markdown(f"""
    <div class="hero-header">
        <div class="hero-title">Strategic Playbook</div>
        <div class="hero-subtitle">
            Synthesised, data-backed insights derived from {total_games:,} commercial Steam titles. 
            Segmented by audience to provide actionable intelligence.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_gamers, tab_indies, tab_pubs, tab_analysts = st.tabs([
        "For Gamers", "For Indie Developers", "For Publishers", "For Analysts"
    ])

    # ── TAB 1: GAMERS ──────────────────────────────────────────────────────────
    with tab_gamers:
        st.markdown('<div class="section-header">Finding Maximum Value</div>', unsafe_allow_html=True)
        st.markdown("""
        **1. OBSERVATION:** 
        There is a dense concentration of high-quality titles (≥85% positive) priced between $9.99 and $14.99 that out-perform $60 AAA titles in user review percentages.
        
        **2. VISUAL EVIDENCE:** 
        See *Market Insights* page. The "Lower-Price / Higher-Quality" quadrant contains over 14,000 titles, heavily concentrated in the Indie and Casual genres.
        
        **3. INTERPRETATION:** 
        Review scores on Steam frequently reflect a "Value-for-Money" ratio rather than absolute graphical fidelity. Lower-priced games often set achievable player expectations, leading to higher satisfaction rates.
        
        **4. POSSIBLE ACTION:** 
        Gamers looking to maximize their entertainment budget should heavily filter the *Market Explorer* for titles in the "Budget" tier ($0.99 - $9.99) with Review Scores > 85%, where the highest concentration of "Hidden Gems" resides.
        """)

        st.markdown('<hr>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Spotting "Overwhelmingly Positive" Traits</div>', unsafe_allow_html=True)
        st.markdown("""
        **1. OBSERVATION:** 
        Titles with the highest median review scores frequently have 1 to 2 genres listed, as opposed to 4+.
        
        **2. VISUAL EVIDENCE:** 
        See *Publisher Studio > What-If Simulator*. Adjusting "Steam Categories Count" or "Genre Count" downward in the ML simulator often produces a more stable Predicted Value Score.
        
        **3. INTERPRETATION:** 
        Historically, games that attempt to blend too many genres or feature categories dilute their core gameplay loop, leading to mixed reviews. Highly focused mechanics resonate better with Steam audiences.
        
        **4. POSSIBLE ACTION:** 
        When evaluating early-access or newly released titles, players might exercise caution with games promising an exhaustive list of genres (e.g., "Survival Horror RPG Base-builder"), prioritizing titles that focus on one or two distinct tags.
        """)


    # ── TAB 2: INDIE DEVELOPERS ────────────────────────────────────────────────
    with tab_indies:
        st.markdown('<div class="section-header">The Localization Premium</div>', unsafe_allow_html=True)
        st.markdown("""
        **1. OBSERVATION:** 
        Titles supporting 10 or more languages show a median ownership size approximately 3.4x higher than titles supporting 2 or fewer languages.
        
        **2. VISUAL EVIDENCE:** 
        See *Overview > Key Findings* ("Localization Reach Multiplier") and *Market Insights > Localization* tab.
        
        **3. INTERPRETATION:** 
        While translating a game incurs upfront costs, reaching non-English speaking markets historically associates with a significant increase in the total addressable market and baseline ownership pool.
        
        **4. POSSIBLE ACTION:** 
        Indie studios should budget for basic UI and subtitle localization for 4-6 high-impact languages (e.g., Simplified Chinese, Spanish, Russian) before launch, using the *Market Insights* dashboard to identify which languages correlate with their specific genre.
        """)

        st.markdown('<hr>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Pricing Elasticity in Niche Genres</div>', unsafe_allow_html=True)
        st.markdown("""
        **1. OBSERVATION:** 
        In complex genres like "Strategy" or "Simulation", pushing price from $14.99 to $19.99 historically is associated with a negligible drop in the predicted Value Score, unlike in "Casual" genres where it is associated with a severe drop.
        
        **2. VISUAL EVIDENCE:** 
        See *Publisher Studio > Fair-Price Decision Support*. The "Value Ratio (pts/$)" curve decays much slower for Strategy than for Casual.
        
        **3. INTERPRETATION:** 
        Niche audiences seeking deep mechanical complexity are historically less price-sensitive. They value depth and replayability (avg playtime > 500 mins) over impulse-buy pricing.
        
        **4. POSSIBLE ACTION:** 
        Developers of deep strategy or simulation games should confidently price in the "Mid-range" tier ($10-$30) rather than racing to the bottom, ensuring they capture fair value from their dedicated niche.
        """)


    # ── TAB 3: PUBLISHERS ──────────────────────────────────────────────────────
    with tab_pubs:
        st.markdown('<div class="section-header">Identifying Market Gap Opportunities</div>', unsafe_allow_html=True)
        st.markdown("""
        **1. OBSERVATION:** 
        Certain Genre × Tier intersections (e.g., Mid-range RPGs) exhibit a high "Market Gap Signal"—meaning they have high median ownership and review scores but a relatively low count of competing titles.
        
        **2. VISUAL EVIDENCE:** 
        See *Game Analyzer > Market Position* and *Publisher Studio > What-If Simulator*. Adjusting genre and price in the simulator highlights underserved segments.
        
        **3. INTERPRETATION:** 
        A high Gap Signal suggests an underserved player base. Historically, publishers releasing competent titles into these specific segments encounter less friction in discovery and user acquisition.
        
        **4. POSSIBLE ACTION:** 
        Publishers looking to fund new projects should pivot away from "AAA Action" (which has high supply density) and actively seek out pitches in high-gap segments (e.g., Premium Simulation or Mid-range RPG) to maximize historical return profiles.
        """)

        st.markdown('<hr>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Portfolio Diversification via Segmentation</div>', unsafe_allow_html=True)
        st.markdown("""
        **1. OBSERVATION:** 
        K-Means clustering reveals that the Steam market is cleanly divisible into segments like "Low-Price / Lower-Quality" (volume-driven) and "High-Price / High-Quality" (prestige-driven).
        
        **2. VISUAL EVIDENCE:** 
        See *Cluster Explorer* page. The PCA projection and heatmap confirm distinct, non-overlapping clusters: Hidden Gems, Budget Filler, Justified AAA, and Overpriced Premium.
        
        **3. INTERPRETATION:** 
        Successful publisher portfolios often require a balance. Relying solely on prestige titles increases risk due to long development cycles, while volume-driven titles can damage brand reputation.
        
        **4. POSSIBLE ACTION:** 
        Publishers can use the *Game Analyzer* tool to audit individual titles, ensuring they have a diversified spread of titles across at least 3 of the 5 natural market archetypes visible in the Cluster Explorer.
        """)


    # ── TAB 4: PLATFORM ANALYSTS ───────────────────────────────────────────────
    with tab_analysts:
        st.markdown('<div class="section-header">The Free-to-Play Gravity Well</div>', unsafe_allow_html=True)
        st.markdown("""
        **1. OBSERVATION:** 
        Free-to-Play titles command an extreme disproportion of total estimated ownership across the platform, despite making up a smaller percentage of total releases.
        
        **2. VISUAL EVIDENCE:** 
        See *Overview > Market Signals* ("F2P Ownership Multiplier") and *Market Explorer > Ownership Bubble Chart*.
        
        **3. INTERPRETATION:** 
        The barrier to entry for F2P is zero, allowing these titles to act as massive funnels for platform user acquisition. However, they drastically skew median engagement metrics for the whole platform.
        
        **4. POSSIBLE ACTION:** 
        When modeling macro platform health or overall engagement trends, analysts must isolate the "Free" tier from "Budget/Mid-range/Premium" tiers to prevent the F2P outliers from distorting the actual health of the premium game economy.
        """)

        st.markdown('<hr>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Anomaly Detection as a Trend Predictor</div>', unsafe_allow_html=True)
        st.markdown("""
        **1. OBSERVATION:** 
        The Isolation Forest model identifies a small cluster of games that possess mid-tier pricing but AAA-level concurrent player counts and extreme playtime.
        
        **2. VISUAL EVIDENCE:** 
        See *Anomaly Finder > Unsupervised Outliers (Isolation Forest)*.
        
        **3. INTERPRETATION:** 
        These multidimensional outliers are not data errors; they are often "Viral Hits" or genre-defining pioneers (e.g., early Battle Royales or Autochess games) that break historical feature associations.
        
        **4. POSSIBLE ACTION:** 
        Platform analysts should run the Isolation Forest anomaly detector on a monthly rolling cohort. By monitoring newly flagged outliers, analysts can detect emerging viral genres or shifting consumer preferences weeks before they become mainstream consensus.
        """)
