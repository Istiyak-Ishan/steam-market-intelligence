"""
validation.py — Data quality checks for Steam Market Intelligence Platform.

All functions accept a DataFrame and return a ValidationResult dict:
  { "ok": bool, "errors": list[str], "warnings": list[str] }
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any

# ── Required columns in the cleaned CSV ──────────────────────────────────────
REQUIRED_COLUMNS = [
    "app_id", "name", "release_date", "price",
    "positive", "negative", "total_review", "review_score_pct",
    "recommendations", "peak_ccu",
    "lowest_estimate_owner", "highest_estimate_owner",
    "average_playtime_forever", "median_playtime_forever",
    "genres", "categories",
    "windows", "mac", "linux", "patforms_count",
    "supported_languages", "languages_count",
    "primary_genre", "genre_count", "categories_count",
    "age_by_years", "price_tier",
]

# ── Model feature columns (must all exist before prediction) ─────────────────
MODEL_FEATURE_COLS = [
    "quality_score", "age_by_years", "categories_count", "languages_count",
    "peak_ccu", "log_reviews", "genre_casual", "genre_count",
    "full_audio_languages_count", "is_indie", "average_playtime_forever",
    "cat_single_player",
]


def _result(errors: list[str], warnings: list[str]) -> dict[str, Any]:
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def validate_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Check that all required columns are present."""
    errors, warnings = [], []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    return _result(errors, warnings)


def validate_no_duplicate_ids(df: pd.DataFrame) -> dict[str, Any]:
    """Warn if app_id is not unique."""
    errors, warnings = [], []
    if "app_id" in df.columns:
        n_dup = df["app_id"].duplicated().sum()
        if n_dup > 0:
            warnings.append(f"{n_dup:,} duplicate app_id values detected.")
    return _result(errors, warnings)


def validate_price_range(df: pd.DataFrame) -> dict[str, Any]:
    """Check for negative or implausibly high prices."""
    errors, warnings = [], []
    if "price" not in df.columns:
        return _result(errors, warnings)
    neg = (df["price"] < 0).sum()
    extreme = (df["price"] > 200).sum()
    if neg > 0:
        errors.append(f"{neg:,} rows have negative price.")
    if extreme > 0:
        warnings.append(f"{extreme:,} rows have price > $200 (possible outliers).")
    return _result(errors, warnings)


def validate_review_score(df: pd.DataFrame) -> dict[str, Any]:
    """Check review_score_pct is within [0, 1]."""
    errors, warnings = [], []
    if "review_score_pct" not in df.columns:
        return _result(errors, warnings)
    col = df["review_score_pct"].dropna()
    out_of_range = ((col < 0) | (col > 1)).sum()
    if out_of_range > 0:
        errors.append(f"{out_of_range:,} review_score_pct values outside [0, 1].")
    return _result(errors, warnings)


def validate_owner_columns(df: pd.DataFrame) -> dict[str, Any]:
    """Check owner estimates are non-negative."""
    errors, warnings = [], []
    for col in ["lowest_estimate_owner", "highest_estimate_owner"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                errors.append(f"{neg:,} negative values in {col}.")
    return _result(errors, warnings)


def validate_missingness(df: pd.DataFrame, threshold: float = 0.50) -> dict[str, Any]:
    """Warn if any column exceeds the missingness threshold."""
    errors, warnings = [], []
    miss_pct = df.isnull().mean()
    high_miss = miss_pct[miss_pct > threshold]
    if not high_miss.empty:
        for col, pct in high_miss.items():
            warnings.append(f"Column '{col}' is {pct*100:.1f}% missing.")
    return _result(errors, warnings)


def validate_model_features(df: pd.DataFrame) -> dict[str, Any]:
    """Check that all model input columns are present in df."""
    errors, warnings = [], []
    missing = [c for c in MODEL_FEATURE_COLS if c not in df.columns]
    if missing:
        errors.append(f"Model features missing from DataFrame: {missing}")
    return _result(errors, warnings)


def run_all_validations(df: pd.DataFrame) -> dict[str, Any]:
    """Run all validation checks and aggregate results."""
    all_errors, all_warnings = [], []
    checks = [
        validate_schema,
        validate_no_duplicate_ids,
        validate_price_range,
        validate_review_score,
        validate_owner_columns,
        validate_missingness,
    ]
    for check in checks:
        result = check(df)
        all_errors.extend(result["errors"])
        all_warnings.extend(result["warnings"])
    return _result(all_errors, all_warnings)
