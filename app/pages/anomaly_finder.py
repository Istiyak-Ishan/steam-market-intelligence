import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.config import SEGMENTATION_FEATURES

@st.cache_data(show_spinner="Computing anomalies...")
def compute_isolation_forest_anomalies(df: pd.DataFrame, contamination=0.02) -> pd.DataFrame:
    df_clean = df.dropna(subset=SEGMENTATION_FEATURES).copy()
    if df_clean.empty:
        return df_clean
        
    X = df_clean[SEGMENTATION_FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X_scaled)
    
    # -1 for anomalies, 1 for normal
    preds = iso.predict(X_scaled)
    # The lower, the more abnormal
    scores = iso.decision_function(X_scaled)
    
    df_clean["is_anomaly"] = preds == -1
    # Convert score so higher is more anomalous
    df_clean["anomaly_score"] = -scores
    
    return df_clean[df_clean["is_anomaly"]].sort_values("anomaly_score", ascending=False)

def render(df):
    st.markdown("<div class='hero-header'><div class='hero-title'>Anomaly Finder</div><div class='hero-subtitle'>Hidden Gems & Viral Breakouts</div></div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Hidden Gems", "Viral Breakouts", "AI Anomalies (Isolation Forest)"])
    
    with tab1:
        st.markdown("<div class='section-header'>Hidden Gems</div>", unsafe_allow_html=True)
        st.info("High quality games with low player counts and high playtime.")
        
        gems = df[
            (df["review_score_pct"] >= 0.90) &
            (df["total_review"] <= 3000) &
            (df["total_review"] >= 50) &
            (df["average_playtime_forever"] >= 120) &
            (df["price"] > 0)
        ].sort_values("review_score_pct", ascending=False)
        
        st.dataframe(
            gems[["name", "primary_genre", "price", "review_score_pct", "total_review", "average_playtime_forever"]],
            use_container_width=True,
            hide_index=True
        )
        
    with tab2:
        st.markdown("<div class='section-header'>Viral Breakouts</div>", unsafe_allow_html=True)
        st.info("Games with massive reach or concurrent player spikes.")
        
        viral = df[
            (df["total_review"] >= 25000) |
            (df["peak_ccu"] >= 10000)
        ].sort_values("peak_ccu", ascending=False)
        
        st.dataframe(
            viral[["name", "primary_genre", "price", "peak_ccu", "total_review", "review_score_pct"]],
            use_container_width=True,
            hide_index=True
        )
        
    with tab3:
        st.markdown("<div class='section-header'>AI Detected Anomalies</div>", unsafe_allow_html=True)
        st.info("Unsupervised Isolation Forest algorithm detects the most statistically unusual games in the market.")
        
        contamination = st.slider("Contamination Rate", 0.001, 0.05, 0.01, 0.005)
        
        anomalies = compute_isolation_forest_anomalies(df, contamination=contamination)
        
        if not anomalies.empty:
            fig = px.scatter(
                anomalies.head(100), 
                x="price", 
                y="anomaly_score",
                color="primary_genre",
                hover_name="name",
                hover_data=["review_score_pct", "total_review"],
                title="Top 100 Anomalies"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Rajdhani", color="#00F0FF")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            cols = ["name", "primary_genre", "anomaly_score"] + SEGMENTATION_FEATURES
            st.dataframe(
                anomalies[cols].head(50),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No anomalies found with this configuration.")
