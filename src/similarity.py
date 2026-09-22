"""
similarity.py — Similar Game Engine for the Steam Market Intelligence Platform.

Uses cosine similarity on a standardised feature matrix.
Returns the N most similar games with scores and comparison attributes.

Excluded from features: app_id, name, URLs, image URLs, identifiers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from src.config import SIMILARITY_FEATURES, SIMILARITY_N_DEFAULT

# ── Module-level cache ─────────────────────────────────────────────────────────
_feature_matrix: np.ndarray | None = None
_scaler_sim: StandardScaler | None = None
_indexed_df: pd.DataFrame | None = None


DISPLAY_COLS = [
    "name", "price", "review_score_pct", "owners_mid",
    "recommendations", "peak_ccu", "average_playtime_forever",
    "languages_count", "patforms_count", "age_by_years",
    "primary_genre", "price_tier",
]


def _build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """
    Build a standardised feature matrix from the dataset.
    Only rows with no NaN in SIMILARITY_FEATURES are used.
    """
    available = [f for f in SIMILARITY_FEATURES if f in df.columns]
    clean = df[available + ["app_id"]].dropna(subset=available).copy()
    scaler = StandardScaler()
    matrix = scaler.fit_transform(clean[available])
    return matrix, scaler, clean.reset_index(drop=True)


def _ensure_matrix(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """Build or return cached feature matrix."""
    global _feature_matrix, _scaler_sim, _indexed_df
    if _feature_matrix is None or _indexed_df is None:
        _feature_matrix, _scaler_sim, _indexed_df = _build_feature_matrix(df)
    return _feature_matrix, _scaler_sim, _indexed_df


def find_similar_games(
    game_profile: dict | pd.Series,
    df: pd.DataFrame,
    n: int = SIMILARITY_N_DEFAULT,
    exclude_app_id: int | None = None,
) -> pd.DataFrame:
    """
    Return the N most similar games to the given profile.

    Parameters
    ----------
    game_profile   : dict or Series with keys matching SIMILARITY_FEATURES
                     (missing values are filled with dataset medians)
    df             : full dataset (from data_loader)
    n              : number of similar games to return
    exclude_app_id : app_id to exclude from results (set when querying an
                     existing game so it doesn't match itself)

    Returns
    -------
    DataFrame with columns from DISPLAY_COLS + 'similarity_score' (0–1)
    """
    matrix, scaler, idx_df = _ensure_matrix(df)
    available = [f for f in SIMILARITY_FEATURES if f in df.columns]

    # Build query vector; fill missing with dataset medians
    medians = df[available].median()
    if isinstance(game_profile, pd.Series):
        game_profile = game_profile.to_dict()

    query_values = []
    for f in available:
        val = game_profile.get(f)
        if pd.isna(val) or val is None:
            val = medians[f]
        query_values.append(float(val))
    
    query_values = np.array(query_values).reshape(1, -1)
    query_scaled = scaler.transform(query_values)

    sims = cosine_similarity(query_scaled, matrix)[0]

    idx_df = idx_df.copy()
    idx_df["similarity_score"] = sims

    if exclude_app_id is not None:
        idx_df = idx_df[idx_df["app_id"] != exclude_app_id]

    top = idx_df.nlargest(n, "similarity_score")[["app_id", "similarity_score"]]

    # Merge back full display info (take only app_id+score from top to avoid column collisions)
    display_cols = [c for c in DISPLAY_COLS if c in df.columns]
    result = top.merge(df[["app_id"] + display_cols].drop_duplicates("app_id"),
                       on="app_id", how="left")
    result["similarity_score"] = result["similarity_score"].round(4)
    return result[display_cols + ["similarity_score", "app_id"]].reset_index(drop=True)


def find_similar_by_appid(
    app_id: int,
    df: pd.DataFrame,
    n: int = SIMILARITY_N_DEFAULT,
) -> pd.DataFrame:
    """Find similar games given an app_id present in df."""
    row = df[df["app_id"] == app_id]
    if row.empty:
        raise ValueError(f"app_id {app_id} not found in dataset.")
    profile = row.iloc[0].to_dict()
    return find_similar_games(profile, df, n=n, exclude_app_id=app_id)


def reset_similarity_cache() -> None:
    """Clear cached similarity matrix (call after reloading data)."""
    global _feature_matrix, _scaler_sim, _indexed_df
    _feature_matrix = _scaler_sim = _indexed_df = None
