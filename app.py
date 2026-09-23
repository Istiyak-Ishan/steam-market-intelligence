"""
Steam Market Intelligence Platform — Main Streamlit Entry Point

Run with:
    streamlit run app.py

Architecture: app.py → app/pages/ + src/
"""
import sys
from pathlib import Path

# Ensure project root is on path so 'src' imports work regardless of CWD
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from src.config import APP_TITLE, APP_ICON, APP_LAYOUT, SIDEBAR_STATE

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state=SIDEBAR_STATE,
)

# ── Design System CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p {
    font-family: 'IBM Plex Sans', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    font-size: 1.25rem;
}

/* ── Color Tokens ──────────────────────────────────────── */
:root {
    /* Backgrounds & Surfaces */
    --bg-base:        #05050a;  /* Deep blue-black */
    --bg-surface:     #0f0f1a;  /* Slightly lighter cool-dark */
    --bg-elevated:    #151525;  
    
    /* Borders & Lines */
    --border-subtle:  rgba(0, 245, 255, 0.15); /* 15% Cyan */
    --border-default: rgba(0, 245, 255, 0.25); /* 25% Cyan */
    --border-active:  rgba(0, 245, 255, 0.5);  /* 50% Cyan */
    
    /* Typography */
    --text-primary:   #ffffff;  /* Bright white for titles */
    --text-secondary: #6b7a99;  /* Muted blue-grey */
    --text-muted:     #4a5568;
    
    /* Accents (Strict Hierarchy) */
    --accent:         #00f5ff;  /* Primary: Electric Cyan */
    --accent-secondary:#ff0066; /* Secondary: Hot Magenta */
    --success:        #00ff88;  /* Neon Green */
    --warning:        #f0ff00;  /* Acid Yellow */
    --danger:         #ff0066;  /* Hot Magenta */
}

/* ── App Background ────────────────────────────────────── */
.stApp {
    background-color: var(--bg-base);
    background-image: 
        linear-gradient(rgba(0, 245, 255, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 245, 255, 0.02) 1px, transparent 1px);
    background-size: 20px 20px;
    background-position: center center;
    color: var(--text-secondary);
}

/* Hide default Streamlit elements except the header (which contains the sidebar toggle) */
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* ── Sidebar ────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-surface);
    border-right: 1px solid var(--border-subtle);
}

/* ── Page Header ─────────────────────────────────────── */
.hero-header {
    padding: 24px 0 16px;
    margin-bottom: 24px;
    border-bottom: 1px solid var(--border-subtle);
}
.hero-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1.45rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.05em;
    text-transform: uppercase;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
    line-height: 1.2;
}
.hero-subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text-muted);
    font-size: 0.82rem;
    margin-top: 5px;
    line-height: 1.5;
}

/* ── Metric Cards ────────────────────────────────────────── */
.metric-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    padding: 16px 20px;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.55rem;
    font-weight: 500;
    color: var(--accent);
    text-shadow: 0 0 8px rgba(0, 245, 255, 0.4); /* Restrained glow */
    line-height: 1.2;
}
.metric-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-top: 6px;
}

/* ── Section Dividers ────────────────────────────────────── */
.section-header {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.11em;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 6px;
    margin: 28px 0 14px 0;
}

/* ── Info Boxes ──────────────────────────────────────────── */
.info-box {
    background: rgba(79, 142, 247, 0.06);
    border-left: 2px solid var(--accent);
    padding: 10px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-bottom: 8px;
}

/* ── Streamlit metric widget ─────────────────────────────── */
div[data-testid="stMetric"] {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-top: 2px solid var(--border-default);
    padding: 14px 16px;
}
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.35rem !important;
    color: var(--accent) !important;
    text-shadow: 0 0 8px rgba(0, 245, 255, 0.4);
}
div[data-testid="stMetric"] label {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.7rem !important;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
}

/* ── Plotly chart container ──────────────────────────────── */
div[data-testid="stPlotlyChart"] {
    background: transparent;
}

/* ── Expanders & Tabs ────────────────────────────────────── */
div[data-testid="stExpander"] {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle) !important;
}
div[data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid var(--border-subtle);
    gap: 2px;
}
button[data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text-muted);
    background: transparent;
    border-bottom: 2px solid transparent !important;
    padding: 8px 14px;
    text-transform: none;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--text-primary) !important;
    background: transparent !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: transparent;
    color: var(--text-secondary);
    border: 1px solid var(--border-default);
    border-radius: 4px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    transition: all 150ms linear;
}
.stButton > button:hover {
    background: var(--bg-elevated);
    color: var(--text-primary);
    border-color: var(--accent);
    box-shadow: 0 0 8px rgba(0, 245, 255, 0.3);
}
.stButton > button:active {
    background: rgba(0, 245, 255, 0.1);
}

/* ── Select boxes and inputs ─────────────────────────────── */
div[data-baseweb="select"] > div, input {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}

/* ── Sidebar Navigation ──────────────────────────────────── */
.nav-brand-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text-primary);
}
.nav-section {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.63rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.11em;
    padding: 12px 0 5px;
}
div[data-testid="stRadio"] > div > label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 400;
    color: var(--text-secondary);
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 5px 8px;
    transition: all 150ms linear;
}
div[data-testid="stRadio"] > div > label:hover {
    background: var(--bg-elevated);
    color: var(--text-primary);
    border-color: var(--border-subtle);
}
div[data-testid="stRadio"] > div > label[data-baseweb="radio"] input:checked + div {
    /* For selected radio background if possible, Streamlit is tricky here 
       but we rely on hover and active states */
}

/* ── Dataframes ───────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-default) !important;
    border-radius: 0px;
}

</style>

""", unsafe_allow_html=True)


# ── Cached data loading ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading Steam dataset…", ttl=3600)
def get_data():
    from src.data_loader import load_data
    from src.feature_engineering import apply_all_features
    from src.validation import run_all_validations
    df = load_data()
    df = apply_all_features(df)
    return df


@st.cache_resource(show_spinner="Loading ML models…")
def get_models():
    from src.model_loader import (
        load_sweetspot_model, load_review_score_model,
        load_value_score_model, load_ownership_model,
        load_price_tier_clf, load_fair_price_clf,
    )
    return {
        "sweetspot":      load_sweetspot_model(),
        "review":         load_review_score_model(),
        "value":          load_value_score_model(),
        "ownership":      load_ownership_model(),
        "price_tier_clf": load_price_tier_clf(),
        "fair_price_clf": load_fair_price_clf(),
    }


# ── Navigation structure ──────────────────────────────────────────────────────
# Format: "Label": "route_key"
# Grouped via section headers rendered separately in sidebar

NAV_GROUPS = [
    ("", [
        ("Overview",               "overview"),
        ("Market Insights",        "market_explorer"),
        ("Genre Benchmark",        "genre_benchmark"),
        ("Anomaly Finder",         "anomaly_finder"),
    ]),
    ("ML Tools", [
        ("Predict Tool",           "predict_tool"),
        ("Feature Importance",     "model_lab"),
        ("Cluster Explorer",       "segmentation"),
    ]),
    ("Game Intelligence", [
        ("Game Analyzer",          "game_analyzer"),
        ("Publisher Studio",       "publisher_studio"),
        ("Strategic Playbook",     "strategic_playbook"),
        ("Game Comparison",        "game_comparison"),
        ("Similar Games",          "similar_games"),
    ]),
    ("", [
        ("EDA Notebook",           "methodology"),
    ]),
]

# Flat list preserving order for st.radio
PAGES_FLAT = []
for _section, items in NAV_GROUPS:
    PAGES_FLAT.extend(items)

PAGE_LABELS = [label for label, _ in PAGES_FLAT]
PAGE_KEYS   = {label: key for label, key in PAGES_FLAT}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div style="padding:20px 4px 16px; border-bottom:1px solid #22252e; margin-bottom:8px;">
        <div class="nav-brand-title">Steam Market Intelligence</div>
        <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.7rem; color:#555e6e; margin-top:3px; text-transform:uppercase; letter-spacing:0.08em;">Data Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # Grouped navigation: each group renders a markdown section header + its own radio.
    # Track selected page in session_state; clicking any group item updates it globally.
    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = PAGE_LABELS[0]

    def _on_nav_change(grp_idx: int) -> None:
        chosen = st.session_state.get(f"nav_grp_{grp_idx}")
        if chosen is not None:
            st.session_state["selected_page"] = chosen
            for i in range(len(NAV_GROUPS)):
                if i != grp_idx:
                    st.session_state[f"nav_grp_{i}"] = None

    for grp_idx, (section, items) in enumerate(NAV_GROUPS):
        if section:
            st.markdown(
                f'<div class="nav-section">{section}</div>',
                unsafe_allow_html=True,
            )
        group_labels = [lbl for lbl, _ in items]
        cur = st.session_state.get("selected_page", PAGE_LABELS[0])
        default_idx = group_labels.index(cur) if cur in group_labels else None

        st.radio(
            f"nav_section_{grp_idx}",
            group_labels,
            index=default_idx,
            label_visibility="collapsed",
            key=f"nav_grp_{grp_idx}",
            on_change=_on_nav_change,
            args=(grp_idx,),
        )

    selected_label = st.session_state.get("selected_page", PAGE_LABELS[0])



page_key = PAGE_KEYS.get(selected_label, "overview")

# ── Pre-load data ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    with st.spinner("Initialising platform…"):
        st.session_state["df"]     = get_data()
        st.session_state["models"] = get_models()

df     = st.session_state["df"]
models = st.session_state["models"]

try:
    if page_key == "overview":
        from app.pages.overview import render
        render(df)

    elif page_key == "market_explorer":
        from app.pages.market_explorer import render
        render(df)

    elif page_key == "predict_tool":
        from app.pages.predict_tool import render
        render(df, models)

    elif page_key == "model_lab":
        from app.pages.model_lab import render
        render(df, models)

    elif page_key == "segmentation":
        from app.pages.segmentation_page import render
        render(df)

    elif page_key == "game_analyzer":
        from app.pages.game_analyzer import render
        render(df, models)

    elif page_key == "publisher_studio":
        from app.pages.publisher_studio import render
        render(df, models)

    elif page_key == "strategic_playbook":
        from app.pages.strategic_playbook import render
        render(df)

    elif page_key == "methodology":
        from app.pages.methodology import render
        render(df)

    elif page_key == "genre_benchmark":
        from app.pages.genre_benchmark import render
        render(df)

    elif page_key == "anomaly_finder":
        from app.pages.anomaly_finder import render
        render(df)

    elif page_key == "game_comparison":
        from app.pages.game_comparison import render
        render(df, models)

    elif page_key == "similar_games":
        from app.pages.similar_games import render
        render(df)

except Exception as e:
    st.error(f"**Application Error:** An unexpected issue occurred while rendering this page.")
    st.error(f"`{str(e)}`")
    st.info("Please adjust your filters or return to the Overview page.")
