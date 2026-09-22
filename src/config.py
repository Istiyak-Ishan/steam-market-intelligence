"""
config.py — Central configuration for Steam Market Intelligence Platform.

All paths are relative to the project root so the project is portable.
All feature lists, model constants and display labels live here.
"""
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR   = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

CLEANED_CSV        = DATA_DIR / "steam_games_cleaned.csv"
GENRE_PIVOT_CSV    = DATA_DIR / "genre_pivot_summary.csv"

REGRESSOR_PKL      = MODELS_DIR / "price_value_regressor.pkl"
CLASSIFIER_PKL     = MODELS_DIR / "price_tier_classifier.pkl"
SCALER_PKL         = MODELS_DIR / "feature_scaler.pkl"

# ── Dataset provenance ────────────────────────────────────────────────────────
RAW_ROW_COUNT      = 136_971   # steam_games.csv shape[0] per preprocessing notebook
CLEANED_ROW_COUNT  = 136_108   # steam_games_cleaned.csv shape[0]
MODELING_ROW_COUNT = 57_685    # commercial titles used for model training (price ≥ 0.99, ≤ 80, ≥5 reviews)
# The EDA notebook loaded steam_games_EDA_ready.csv and filtered primary_genre≥50 games → ~126k
# That explains the "126,130" figure in the executive report: it is a genre-filtered EDA subset.

# ── Model feature columns (exact order as trained) ───────────────────────────
BASE_FEATURES = [
    "age_by_years",               # game age in fractional years
    "categories_count",           # number of Steam categories
    "languages_count",            # number of supported languages
    "peak_ccu",                   # peak concurrent users
    "log_reviews",                # log1p(total_review)
    "genre_casual",               # 1 if genres contains 'Casual'
    "genre_count",                # number of listed genres
    "full_audio_languages_count", # number of full-audio languages
    "is_indie",                   # 1 if genres contains 'Indie'
    "average_playtime_forever",   # average playtime (minutes)
    "cat_single_player",          # 1 if categories contains 'Single-player'
]

# ── Price tier definitions (matching notebook bins) ───────────────────────────
PRICE_TIER_BINS   = [-1, 0, 10, 30, 60, float("inf")]
PRICE_TIER_LABELS = ["Free", "Budget", "Mid-range", "Premium", "AAA"]

# Tiers present in the classifier (Free games were excluded from training)
CLASSIFIER_TIERS  = ["Budget", "Mid-range", "Premium", "AAA"]

# ── owners_mid calculation ─────────────────────────────────────────────────────
# owners_mid = (lowest_estimate_owner + highest_estimate_owner) / 2

# ── value_score (notebook definition) ─────────────────────────────────────────
# value_score_calc = quality_score / price_usd   (quality pts per $1)
# value_score (in cleaned CSV) = metacritic_score / price  — different, often inf
# We always use value_score_calc in the app.

# ── Genre analysis config ─────────────────────────────────────────────────────
MIN_GENRE_GAMES = 50    # minimum games for a genre to appear in analyses

# Top genres to highlight (ordered by game count in dataset)
PRIMARY_GENRES = [
    "Action", "Casual", "Adventure", "Indie", "Simulation",
    "RPG", "Strategy", "Racing", "Sports",
    "Free To Play", "Massively Multiplayer",
]

# ── Similarity engine config ──────────────────────────────────────────────────
SIMILARITY_FEATURES = [
    "price",
    "review_score_pct",
    "owners_mid",
    "recommendations",
    "peak_ccu",
    "average_playtime_forever",
    "languages_count",
    "patforms_count",
    "age_by_years",
]

SIMILARITY_N_DEFAULT = 10   # default number of similar games to return

# ── Segmentation config ───────────────────────────────────────────────────────
SEGMENTATION_FEATURES = [
    "price",
    "review_score_pct",
    "owners_mid",
    "recommendations",
    "peak_ccu",
    "average_playtime_forever",
    "age_by_years",
    "patforms_count",
    "languages_count",
]
SEGMENTATION_K_RANGE  = range(3, 9)   # k values to test
SEGMENTATION_K_DEFAULT = 5             # fallback default

# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE  = "plotly_dark"
PLOTLY_BG_COLOR  = "rgba(0,0,0,0)"   # Transparent for HUD blending
PLOTLY_PAPER_BG  = "rgba(0,0,0,0)"   # Transparent for HUD blending
PLOTLY_FONT_COLOR = "#00F0FF"        # Electric Blue text for tactical feel
ACCENT_COLORS    = [
    "#FF4500",  # Tactical Orange (primary)
    "#00F0FF",  # Electric Blue
    "#39FF14",  # Neon Green
    "#FF003C",  # Crimson
    "#F59E0B",  # Amber
    "#8B5CF6",  # Violet (retained for specific needs)
    "#E2E8F0",  # Silver
    "#10B981",  # Emerald
]

# ── Market-position percentile thresholds ────────────────────────────────────
# Used for descriptive quadrant labels (no value judgement intended).
MARKET_POSITION_PRICE_MED   = 0.50   # median split
MARKET_POSITION_QUALITY_MED = 0.50   # median split

# ── Streamlit app config ──────────────────────────────────────────────────────
APP_TITLE       = "Steam Market Intelligence Platform"
APP_ICON        = "🎮"
APP_LAYOUT      = "wide"
SIDEBAR_STATE   = "expanded"
