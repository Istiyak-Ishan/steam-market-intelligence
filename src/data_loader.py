"""
data_loader.py — Dataset loading and type-safe access for the platform.

Key design decisions:
  - Only reads from data/ directory using relative paths from config.
  - Returns a clean copy; never mutates the cached frame.
  - On first call applies all type conversions needed for downstream modules.
  - The 'steam_games_EDA_ready.csv' referenced in EDA.ipynb does not exist;
    we rebuild equivalent features from steam_games_cleaned.csv.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CLEANED_CSV, GENRE_PIVOT_CSV, MIN_GENRE_GAMES

log = logging.getLogger(__name__)

# ── Internal cache (module-level, not Streamlit) ─────────────────────────────
_df_cache: pd.DataFrame | None = None


def _load_raw() -> pd.DataFrame:
    """Load and type-coerce steam_games_cleaned.csv exactly once."""
    path = Path(CLEANED_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Place steam_games_cleaned.csv in the data/ directory."
        )
    log.info(f"Loading dataset from {path} …")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")

    # ── Numeric coercions ────────────────────────────────────────────────────
    num_cols = [
        "price", "positive", "negative", "total_review", "review_score_pct",
        "recommendations", "peak_ccu", "metacritic_score",
        "average_playtime_forever", "median_playtime_forever",
        "average_playtime_2weeks", "median_playtime_2weeks",
        "achievements", "dlc_count",
        "lowest_estimate_owner", "highest_estimate_owner",
        "windows", "mac", "linux", "patforms_count",
        "languages_count", "genre_count", "categories_count",
        "age_by_years", "release_quarter",
        "has_english", "playtime_engagment_ratio",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Boolean columns ───────────────────────────────────────────────────────
    for col in ["has_reviews", "has_achievements"]:
        if col in df.columns and df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )

    # ── Release year ─────────────────────────────────────────────────────────
    if "release_date" in df.columns:
        df["release_year"] = pd.to_datetime(
            df["release_date"], errors="coerce"
        ).dt.year.astype("Int64")

    # ── owners_mid ───────────────────────────────────────────────────────────
    if "owners_mid" not in df.columns:
        df["owners_mid"] = (
            df["lowest_estimate_owner"] + df["highest_estimate_owner"]
        ) / 2

    # ── Strip junk from primary_genre & normalize ────────────────────────────
    if "primary_genre" in df.columns:
        df["primary_genre"] = df["primary_genre"].astype(str).str.strip()
        df.loc[df["primary_genre"].str.startswith("{"), "primary_genre"] = np.nan
        df.loc[df["primary_genre"] == "nan", "primary_genre"] = np.nan

        # Fallback to first genre listed in 'genres' if primary_genre is missing
        if "genres" in df.columns:
            fallback = df["genres"].dropna().astype(str).str.split(",").str[0].str.strip()
            df["primary_genre"] = df["primary_genre"].fillna(fallback)
            df.loc[df["primary_genre"] == "nan", "primary_genre"] = np.nan

        # Normalize localized translation aliases
        alias_map = {
            "Aventura": "Adventure",
            "Aventure": "Adventure",
            "インディー": "Indie",
        }
        df["primary_genre"] = df["primary_genre"].replace(alias_map)

    # ── full_audio_languages_count ───────────────────────────────────────────
    if "full_audio_languages_count" not in df.columns and "full_audio_languages" in df.columns:
        df["full_audio_languages_count"] = (
            df["full_audio_languages"]
            .fillna("")
            .apply(lambda x: len([s for s in x.split(",") if s.strip()]) if x.strip() else 0)
        )

    return df


def load_data(force_reload: bool = False) -> pd.DataFrame:
    """Return the cleaned dataset.  Cached after first load."""
    global _df_cache
    if _df_cache is None or force_reload:
        _df_cache = _load_raw()
    return _df_cache.copy()


def load_genre_pivot() -> pd.DataFrame:
    """Load the pre-computed genre_pivot_summary.csv."""
    path = Path(GENRE_PIVOT_CSV)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_main_genre_df(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return rows where primary_genre has at least MIN_GENRE_GAMES games."""
    if df is None:
        df = load_data()
    counts = df["primary_genre"].value_counts()
    valid_genres = counts[counts >= MIN_GENRE_GAMES].index
    return df[df["primary_genre"].isin(valid_genres)].copy()
