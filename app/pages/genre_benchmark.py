import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.analytics import genre_summary
from src.market_position import competition_genre_x_tier, compute_market_gap_signal, release_density_by_genre_year

def render(df):
    st.markdown("<div class='hero-header'><div class='hero-title'>Genre Benchmark</div><div class='hero-subtitle'>Market Landscape Analysis</div></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>Genre Summaries</div>", unsafe_allow_html=True)
    g_sum = genre_summary(df, min_games=20)
    st.dataframe(g_sum, use_container_width=True, hide_index=True)
    
    st.markdown("<div class='section-header'>Market Gap Signals</div>", unsafe_allow_html=True)
    gaps = compute_market_gap_signal(df, min_games=30)
    
    fig_gaps = px.bar(
        gaps, 
        x="primary_genre", 
        y="gap_signal_norm",
        color="gap_signal_norm",
        color_continuous_scale="Viridis",
        title="Market Opportunity (Gap Signal)"
    )
    fig_gaps.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)", 
        font=dict(color="#00F0FF", family="Rajdhani")
    )
    st.plotly_chart(fig_gaps, use_container_width=True)

    st.markdown("<div class='section-header'>Competition by Tier</div>", unsafe_allow_html=True)
    pivot_tier = competition_genre_x_tier(df, min_genre_games=30)
    st.dataframe(pivot_tier, use_container_width=True)

    st.markdown("<div class='section-header'>Release Density over Time</div>", unsafe_allow_html=True)
    density = release_density_by_genre_year(df, top_genres=5)
    
    if not density.empty:
        fig_density = px.line(
            density,
            title="Releases by Top Genres Over Time",
            markers=True
        )
        fig_density.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font=dict(color="#00F0FF", family="Rajdhani")
        )
        st.plotly_chart(fig_density, use_container_width=True)
