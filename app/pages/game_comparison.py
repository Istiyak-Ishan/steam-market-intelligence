import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.config import ACCENT_COLORS

PRESET_SHOWDOWNS = {
    "RPG Titans": ["The Witcher 3: Wild Hunt", "Cyberpunk 2077", "Skyrim"],
    "Indie Darlings": ["Stardew Valley", "Hades", "Terraria"],
    "Multiplayer Hits": ["Apex Legends", "Counter-Strike 2", "Dota 2"]
}

def _extract_model_profile(row: pd.Series) -> dict:
    # Build a dictionary formatted for predict_value_score and predict_price_tier
    # Just passing row.to_dict() works as long as apply_all_features has run,
    # because feature_engineering.build_model_input will pluck out what it needs.
    return row.to_dict()

def _render_multi_radar(sample_rows, medians):
    fig = go.Figure()
    
    radar_cols = ["price", "review_score_pct", "total_review", "peak_ccu", 
                  "average_playtime_forever", "languages_count", "patforms_count"]
    
    for i, row in enumerate(sample_rows):
        color = ACCENT_COLORS[i % len(ACCENT_COLORS)]
        
        # Normalize against medians for visualization
        normalized_vals = []
        for col in radar_cols:
            val = float(row.get(col, 0))
            med = float(medians.get(col, 1))
            if med == 0:
                med = 1.0
            # Cap at 2.5x median for radar scale
            norm = min(val / med, 2.5)
            normalized_vals.append(norm)
            
        # Close the radar loop
        normalized_vals.append(normalized_vals[0])
        labels = radar_cols + [radar_cols[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_vals,
            theta=labels,
            fill='toself',
            name=row.get("name", f"Game {i+1}"),
            line=dict(color=color),
            fillcolor=color.replace(')', ', 0.2)').replace('rgb', 'rgba') if 'rgb' in color else color + '33'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 2.5], tickfont=dict(color="#4a7b93")),
            angularaxis=dict(tickfont=dict(color="#00F0FF"))
        ),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Rajdhani", color="#00F0FF")
    )
    return fig

def render(df, models=None):
    st.markdown("<div class='hero-header'><div class='hero-title'>Game Comparison</div><div class='hero-subtitle'>Head-to-Head Analysis</div></div>", unsafe_allow_html=True)
    
    showdown_name = st.selectbox("Select Preset Showdown (or create your own)", list(PRESET_SHOWDOWNS.keys()) + ["<Custom>"])
    
    if showdown_name == "<Custom>":
        default_titles = []
    else:
        default_titles = [t for t in PRESET_SHOWDOWNS[showdown_name] if t in df["name"].values]
        
    titles = st.multiselect(
        "Select Games to Compare (max 5)", 
        options=df["name"].dropna().unique(), 
        default=default_titles,
        max_selections=5
    )
    
    # Filter dataset for these games
    games_df = df[df["name"].isin(titles)]
    
    if games_df.empty:
        st.warning("Please select at least one game to compare.")
        return
        
    st.markdown("<div class='section-header'>Comparison Radar</div>", unsafe_allow_html=True)
    
    radar_cols = ["price", "review_score_pct", "total_review", "peak_ccu", 
                  "average_playtime_forever", "languages_count", "patforms_count"]
    medians = df[radar_cols].median()
    
    sample_rows = games_df.to_dict(orient="records")
    fig = _render_multi_radar(sample_rows, medians)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<div class='section-header'>Game Stats</div>", unsafe_allow_html=True)
    
    cols = st.columns(len(sample_rows))
    for i, row in enumerate(sample_rows):
        with cols[i]:
            st.markdown(f"### {row['name']}")
            st.metric("Price", f"${row['price']:.2f}" if pd.notna(row.get('price')) else "N/A")
            st.metric("Review Score", f"{row['review_score_pct']*100:.1f}%" if pd.notna(row.get('review_score_pct')) else "N/A")
            st.metric("Total Reviews", f"{row['total_review']:,.0f}" if pd.notna(row.get('total_review')) else "N/A")
            
            recs = row.get("recommendations", 0)
            st.metric("Recommendations", f"{recs:,.0f}" if pd.notna(recs) else "0")
            
            st.metric("Peak CCU", f"{row['peak_ccu']:,.0f}" if pd.notna(row.get('peak_ccu')) else "N/A")
            
            if models:
                from src.model_loader import predict_value_score, predict_price_tier
                profile = _extract_model_profile(pd.Series(row))
                try:
                    val_score = predict_value_score(profile)
                    tier, _ = predict_price_tier(profile)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"**Predicted Value Score:** {val_score:.2f}")
                    st.markdown(f"**Predicted Tier:** {tier}")
                except Exception:
                    pass
