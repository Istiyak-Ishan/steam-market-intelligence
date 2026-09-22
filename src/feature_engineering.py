"""
feature_engineering.py — Derive all model-ready and analytical features.

Every function documents:
  - source columns
  - transformation
  - resulting meaning

These mirror the exact definitions used in the original notebooks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Core review features ──────────────────────────────────────────────────────

def add_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    quality_score = review_score_pct × 100
    Source: review_score_pct (already in cleaned CSV)
    Meaning: percentage of positive reviews as a 0–100 float
    """
    df = df.copy()
    df["quality_score"] = (df["review_score_pct"] * 100).round(2)
    return df


def add_log_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    log_reviews = log1p(total_review)
    Source: total_review = positive + negative
    Meaning: log-transformed review count to reduce skew
    """
    df = df.copy()
    df["log_reviews"] = np.log1p(df["total_review"].fillna(0))
    return df


def add_log_owners(df: pd.DataFrame) -> pd.DataFrame:
    """
    log_owners = log1p(owners_mid)
    Source: owners_mid = (lowest_estimate_owner + highest_estimate_owner) / 2
    Meaning: log-transformed ownership midpoint
    """
    df = df.copy()
    if "owners_mid" not in df.columns:
        df["owners_mid"] = (
            df["lowest_estimate_owner"] + df["highest_estimate_owner"]
        ) / 2
    df["log_owners"] = np.log1p(df["owners_mid"].fillna(0))
    return df


# ── Price features ────────────────────────────────────────────────────────────

def add_price_usd(df: pd.DataFrame) -> pd.DataFrame:
    """
    price_usd = expm1(price)  [only used in model training notebook]
    NOTE: In the cleaned CSV, 'price' is already in USD dollars (not log-transformed).
    The model training notebook applied expm1 to the EDA-ready CSV which had
    log-transformed prices. The cleaned CSV price is the raw dollar value.
    For the application we use 'price' directly.
    """
    df = df.copy()
    if "price_usd" not in df.columns:
        df["price_usd"] = df["price"]
    return df


def add_price_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    price_tier: Free | Budget | Mid-range | Premium | AAA
    Source: price
    Bins: [-1, 0] → Free, (0, 10] → Budget, (10, 30] → Mid-range,
          (30, 60] → Premium, (60, ∞) → AAA
    Matches preprocessing notebook definition exactly.
    """
    df = df.copy()
    if "price_tier" not in df.columns:
        df["price_tier"] = pd.cut(
            df["price"],
            bins=[-1, 0, 10, 30, 60, np.inf],
            labels=["Free", "Budget", "Mid-range", "Premium", "AAA"],
        ).astype(str)
    return df


# ── Value score ───────────────────────────────────────────────────────────────

def add_value_score_calc(df: pd.DataFrame) -> pd.DataFrame:
    """
    value_score_calc = quality_score / price_usd   (for paid games only)
    Source: quality_score, price
    Meaning: quality points per $1 spent — the model target
    NOTE: The cleaned CSV has a 'value_score' column = metacritic_score / price,
    which is different and often inf. The modeling notebooks use value_score_calc.
    """
    df = df.copy()
    if "quality_score" not in df.columns:
        df = add_quality_score(df)
    price_col = df.get("price_usd", df["price"])
    df["value_score_calc"] = np.where(
        (price_col > 0) & (df["quality_score"].notna()),
        (df["quality_score"] / price_col).round(2),
        np.nan,
    )
    return df


# ── Genre/category binary flags ───────────────────────────────────────────────

def add_genre_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    genre_casual, genre_action, genre_adventure, genre_rpg,
    genre_simulation, genre_strategy, is_indie
    Source: genres (comma-separated string)
    Meaning: 1/0 binary membership indicators
    """
    df = df.copy()
    genres_str = df["genres"].fillna("")
    genre_map = {
        "genre_casual":     "Casual",
        "genre_action":     "Action",
        "genre_adventure":  "Adventure",
        "genre_rpg":        "RPG",
        "genre_simulation": "Simulation",
        "genre_strategy":   "Strategy",
        "is_indie":         "Indie",
    }
    for col, keyword in genre_map.items():
        if col not in df.columns:
            df[col] = genres_str.str.contains(keyword, case=False, na=False).astype(int)
    return df


def add_category_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    cat_single_player, cat_multi_player, cat_co_op, cat_full_controller_support
    Source: categories (comma-separated string)
    Meaning: 1/0 binary membership indicators
    """
    df = df.copy()
    cats_str = df["categories"].fillna("")
    cat_map = {
        "cat_single_player":           "Single-player",
        "cat_multi_player":            "Multi-player",
        "cat_co_op":                   "Co-op",
        "cat_full_controller_support": "Full controller support",
    }
    for col, keyword in cat_map.items():
        if col not in df.columns:
            df[col] = cats_str.str.contains(keyword, case=False, na=False).astype(int)
    return df


# ── Platform / language features ──────────────────────────────────────────────

def add_platform_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    patforms_count (preserves the original spelling typo from preprocessing)
    Source: windows + mac + linux (binary 0/1)
    Meaning: number of supported platforms
    """
    df = df.copy()
    if "patforms_count" not in df.columns:
        df["patforms_count"] = df[["windows", "mac", "linux"]].sum(axis=1)
    return df


def add_language_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    languages_count = len(supported_languages.split(','))
    Source: supported_languages
    Meaning: number of supported display languages
    Already in the cleaned CSV; recalculates only if missing.
    """
    df = df.copy()
    if "languages_count" not in df.columns and "supported_languages" in df.columns:
        df["languages_count"] = (
            df["supported_languages"]
            .fillna("")
            .str.split(",")
            .apply(len)
        )
    return df


def add_full_audio_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    full_audio_languages_count
    Source: full_audio_languages (comma-separated string or 'None')
    Meaning: number of languages with full audio support
    """
    df = df.copy()
    if "full_audio_languages_count" not in df.columns:
        def _count(x):
            if not x or str(x).strip() in ("", "None", "nan"):
                return 0
            return len([s for s in str(x).split(",") if s.strip()])
        df["full_audio_languages_count"] = (
            df["full_audio_languages"].fillna("").apply(_count)
        )
    return df


# ── Engagement / playtime features ───────────────────────────────────────────

def add_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    playtime_engagment_ratio (from cleaned CSV, already computed)
    recent_vs_lifetime_engagment (from cleaned CSV)
    average_playtime_hours: average_playtime_forever / 60
    Meaning: playtime and engagement intensity measures
    """
    df = df.copy()
    # Already in cleaned CSV; add hours convenience column
    if "average_playtime_hours" not in df.columns:
        df["average_playtime_hours"] = (
            df["average_playtime_forever"].fillna(0) / 60
        ).round(2)
    if "median_playtime_hours" not in df.columns:
        df["median_playtime_hours"] = (
            df["median_playtime_forever"].fillna(0) / 60
        ).round(2)
    return df


# ── Time features ─────────────────────────────────────────────────────────────

def add_release_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    release_year extracted from release_date.
    Already computed by data_loader; idempotent here.
    """
    df = df.copy()
    if "release_year" not in df.columns and "release_date" in df.columns:
        df["release_year"] = pd.to_datetime(
            df["release_date"], errors="coerce"
        ).dt.year.astype("Int64")
    return df


# ── Master pipeline ───────────────────────────────────────────────────────────

def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply every feature engineering step in the correct order.
    Safe to call on the cleaned CSV output from data_loader.load_data().
    """
    df = add_price_usd(df)
    df = add_price_tier(df)
    df = add_quality_score(df)
    df = add_log_reviews(df)
    df = add_log_owners(df)
    df = add_value_score_calc(df)
    df = add_genre_flags(df)
    df = add_category_flags(df)
    df = add_platform_count(df)
    df = add_language_count(df)
    df = add_full_audio_count(df)
    df = add_engagement_features(df)
    df = add_release_year(df)
    return df


def build_model_input(game_profile: dict) -> pd.DataFrame:
    """
    Construct a single-row DataFrame in the exact feature order required by
    the trained models. All missing values are filled with 0.

    game_profile keys (all optional, default to 0):
        quality_score, age_by_years, categories_count, languages_count,
        peak_ccu, log_reviews, genre_casual, genre_count,
        full_audio_languages_count, is_indie, average_playtime_forever,
        cat_single_player
    """
    from src.config import MODEL_FEATURES
    row = {f: float(game_profile.get(f, 0)) for f in MODEL_FEATURES}
    return pd.DataFrame([row])[MODEL_FEATURES]
