import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.similarity import find_similar_games
from src.config import ACCENT_COLORS



def _render_comparison_radar(target: dict, comps: list, medians: pd.Series):
    fig = go.Figure()
    
    radar_cols = ["price", "review_score_pct", "total_review", "peak_ccu", 
                  "average_playtime_forever", "languages_count", "patforms_count"]
    
    # Render comps
    for i, comp in enumerate(comps):
        normalized_vals = []
        for col in radar_cols:
            val = float(comp.get(col, 0))
            med = float(medians.get(col, 1))
            if med == 0:
                med = 1.0
            norm = min(val / med, 2.5)
            normalized_vals.append(norm)
            
        normalized_vals.append(normalized_vals[0])
        labels = radar_cols + [radar_cols[0]]
        
        color = ACCENT_COLORS[(i + 1) % len(ACCENT_COLORS)]
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_vals,
            theta=labels,
            fill='none',
            name=comp.get("name", f"Comp {i+1}"),
            line=dict(color=color, dash='dot')
        ))
        
    # Render target (more prominent)
    target_norm = []
    for col in radar_cols:
        val = float(target.get(col, 0))
        med = float(medians.get(col, 1))
        if med == 0:
            med = 1.0
        norm = min(val / med, 2.5)
        target_norm.append(norm)
        
    target_norm.append(target_norm[0])
    
    fig.add_trace(go.Scatterpolar(
        r=target_norm,
        theta=labels,
        fill='toself',
        name=target.get("name", "Target"),
        line=dict(color=ACCENT_COLORS[0], width=3),
        fillcolor=ACCENT_COLORS[0].replace(')', ', 0.3)').replace('rgb', 'rgba') if 'rgb' in ACCENT_COLORS[0] else ACCENT_COLORS[0] + '4D'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 2.5], tickfont=dict(color="#555e6e")),
            angularaxis=dict(tickfont=dict(color="#9da3ae"))
        ),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans", color="#9da3ae")
    )
    return fig

def render(df):
    st.markdown("<div class='hero-header'><div class='hero-title'>Similar Games</div><div class='hero-subtitle'>Find Competitors & Inspiration</div></div>", unsafe_allow_html=True)
    
    popular_games = df.sort_values("total_review", ascending=False)["name"].dropna().head(15000).tolist()
    target_name = st.selectbox(
        "Search game name (Top 15k most reviewed)", 
        options=popular_games, 
        index=None, 
        placeholder="e.g. Cyberpunk 2077, Stardew Valley…"
    )
    
    if not target_name:
        return
        
    target_row = df[df["name"] == target_name]
    if target_row.empty:
        st.warning("Game not found in dataset.")
        return
        
    target_dict = target_row.iloc[0].to_dict()
    
    st.markdown(f"### Target: {target_name}")
    
    with st.spinner("Finding similar games..."):
        sim_df = find_similar_games(target_dict, df, n=8)
        
    st.markdown("<div class='section-header'>Top Competitors</div>", unsafe_allow_html=True)
    st.dataframe(
        sim_df[["name", "primary_genre", "price", "review_score_pct", "similarity_score"]], 
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("<div class='section-header'>Feature Comparison</div>", unsafe_allow_html=True)
    radar_cols = ["price", "review_score_pct", "total_review", "peak_ccu", 
                  "average_playtime_forever", "languages_count", "patforms_count"]
    medians = df[radar_cols].median()
    
    comps = sim_df.head(3).to_dict(orient="records")
    fig = _render_comparison_radar(target_dict, comps, medians)
    st.plotly_chart(fig, use_container_width=True)
