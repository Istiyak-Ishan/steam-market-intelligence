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
    font-family: 'Rajdhani', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    font-size: 1.1rem;
}

/* ── Color Tokens ──────────────────────────────────────── */
:root {
    --bg-base:        #040508;
    --bg-surface:     #080a0f;
    --bg-elevated:    rgba(10, 14, 23, 0.75);
    --bg-overlay:     rgba(0, 240, 255, 0.05);
    --border-subtle:  #152033;
    --border-default: #1e3a5f;
    --border-strong:  #00F0FF;
    --text-primary:   #00F0FF;
    --text-secondary: #7bb5cf;
    --text-muted:     #4a7b93;
    --accent-violet:  #FF4500;
    --accent-blue:    #00F0FF;
}

/* ── App Background ────────────────────────────────────── */
.stApp {
    background-color: var(--bg-base);
    background-image: 
        linear-gradient(rgba(0, 240, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 240, 255, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    color: var(--text-primary);
}

header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* ── Sidebar ────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-surface);
    border-right: 1px solid var(--border-default);
    box-shadow: inset -5px 0 20px rgba(0, 240, 255, 0.03);
}

/* ── Hero Header ─────────────────────────────────────────── */
.hero-header {
    background: rgba(0, 240, 255, 0.03);
    border: 1px solid var(--accent-blue);
    border-radius: 0px;
    padding: 24px 30px 20px;
    margin-bottom: 24px;
    position: relative;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.1), inset 0 0 20px rgba(0, 240, 255, 0.05);
    clip-path: polygon(0 0, 100% 0, 100% calc(100% - 15px), calc(100% - 15px) 100%, 0 100%);
}
.hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 100%; height: 2px;
    background: var(--accent-blue);
    box-shadow: 0 0 15px var(--accent-blue);
}
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent-blue);
    text-transform: uppercase;
    text-shadow: 0 0 10px rgba(0, 240, 255, 0.4);
    letter-spacing: 2px;
}
.hero-subtitle {
    font-family: 'Rajdhani', sans-serif;
    color: var(--text-secondary);
    font-size: 1.15rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Metric Cards ────────────────────────────────────────── */
.metric-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: 0px;
    padding: 16px 20px;
    margin-bottom: 10px;
    position: relative;
    border-left: 3px solid var(--accent-violet);
    box-shadow: inset 0 0 15px rgba(0, 0, 0, 0.6);
}
.metric-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.8rem;
    color: var(--accent-violet);
    text-shadow: 0 0 10px rgba(255, 69, 0, 0.4);
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 8px;
}

/* ── Section Headers ─────────────────────────────────────── */
.section-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.1rem;
    color: var(--accent-blue);
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid var(--border-default);
    padding-bottom: 5px;
    margin: 25px 0 15px 0;
    text-shadow: 0 0 5px rgba(0, 240, 255, 0.3);
}

/* ── Info Boxes ──────────────────────────────────────────── */
.info-box {
    background: rgba(255, 69, 0, 0.05);
    border: 1px solid var(--accent-violet);
    border-radius: 0px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: var(--accent-violet);
    clip-path: polygon(10px 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%, 0 10px);
}

/* ── Streamlit metric widget ─────────────────────────────── */
div[data-testid="stMetric"] {
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: 0px;
    border-top: 2px solid var(--accent-blue);
    padding: 14px 16px;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.6);
}
div[data-testid="stMetricValue"] {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--accent-blue) !important;
    text-shadow: 0 0 8px rgba(0, 240, 255, 0.3);
}
div[data-testid="stMetric"] label {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Plotly chart container ──────────────────────────────── */
div[data-testid="stPlotlyChart"] {
    border: 1px solid var(--border-default);
    border-radius: 0px;
    background: var(--bg-elevated);
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.03);
}

/* ── Expanders \u0026 Tabs ───────────────────────────────────────────── */
div[data-testid="stExpander"] {
    background: var(--bg-elevated);
    border: 1px solid var(--border-default) !important;
    border-radius: 0px;
}
div[data-baseweb="tab-list"] {
    background: var(--bg-elevated);
    border-radius: 0px;
    border: 1px solid var(--border-default);
    border-bottom: 2px solid var(--accent-blue);
}
button[data-baseweb="tab"] {
    font-family: 'Orbitron', sans-serif;
    text-transform: uppercase;
    color: var(--text-muted);
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(0, 240, 255, 0.1) !important;
    color: var(--accent-blue) !important;
    text-shadow: 0 0 5px rgba(0, 240, 255, 0.5);
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: transparent;
    color: var(--accent-blue);
    border: 1px solid var(--accent-blue);
    border-radius: 0px;
    font-family: 'Orbitron', sans-serif;
    text-transform: uppercase;
    letter-spacing: 1px;
    clip-path: polygon(10px 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%, 0 10px);
}
.stButton > button:hover {
    background: rgba(0, 240, 255, 0.15);
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.3);
    color: #fff;
    border-color: #fff;
}

/* ── Select boxes and inputs ─────────────────────────────── */
div[data-baseweb="select"] > div, input {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 0px !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar Nav ───────────────────────────────────── */
.nav-section {
    font-family: 'Orbitron', sans-serif;
    color: var(--accent-violet);
    text-shadow: 0 0 5px rgba(255, 69, 0, 0.3);
}
.nav-brand-title {
    font-family: 'Orbitron', sans-serif;
    color: var(--accent-blue);
}
div[data-testid="stRadio"] > div > label {
    font-family: 'Rajdhani', sans-serif;
    text-transform: uppercase;
    border-radius: 0px;
}
div[data-testid="stRadio"] > div > label:hover {
    background: rgba(0, 240, 255, 0.1);
    color: var(--accent-blue);
    border-left: 2px solid var(--accent-blue);
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
        ("🏠  Overview",          "overview"),
        ("♟️  Strategic Playbook", "strategic_playbook"),
    ]),
    ("MARKET INTELLIGENCE", [
        ("📊  Market Explorer",   "market_explorer"),
    ]),
    ("GAME INTELLIGENCE", [
        ("🎮  Game Analyzer",     "game_analyzer"),
        ("⚔️  Competitive Analysis", "competitive_analysis"),
    ]),
    ("CREATOR TOOLS", [
        ("🚀  Publisher Studio",  "publisher_studio"),
        ("🧪  Advanced Analytics","advanced_analytics"),
    ]),
    ("", [
        ("📖  EDA",               "methodology"),
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
    <div class="nav-brand">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
            <div style="width:32px;height:32px;background:linear-gradient(135deg,#7c6af7,#4f8ef7);
                        border-radius:8px;display:flex;align-items:center;justify-content:center;
                        font-size:16px;">🎮</div>
            <div>
                <div class="nav-brand-title">Steam Market Intelligence</div>
                <div class="nav-brand-sub">Data Analytics Platform</div>
            </div>
        </div>
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

    elif page_key == "strategic_playbook":
        from app.pages.strategic_playbook import render
        render(df)

    elif page_key == "market_explorer":
        from app.pages.market_explorer import render
        render(df)

    elif page_key == "game_analyzer":
        from app.pages.game_analyzer import render
        render(df, models)

    elif page_key == "competitive_analysis":
        from app.pages.competitive_analysis import render
        render(df, models)

    elif page_key == "publisher_studio":
        from app.pages.publisher_studio import render
        render(df, models)

    elif page_key == "advanced_analytics":
        from app.pages.advanced_analytics import render
        render(df, models)

    elif page_key == "methodology":
        from app.pages.methodology import render
        render(df)
        
except Exception as e:
    st.error(f"**Application Error:** An unexpected issue occurred while rendering this page.")
    st.error(f"`{str(e)}`")
    st.info("Please adjust your filters or return to the Overview page.")
