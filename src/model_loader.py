"""
model_loader.py — Safe model loading layer for the platform.

Usage:
    from src.model_loader import load_price_value_model, load_price_tier_model, load_scaler

Each function returns the model or raises a descriptive ModelLoadError.
Models are cached at module level.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import REGRESSOR_PKL, CLASSIFIER_PKL, SCALER_PKL, MODEL_FEATURES

log = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when a model artifact cannot be loaded or used."""


def _safe_load(path: Path, label: str) -> Any:
    """Load a joblib artifact, capturing sklearn version warnings."""
    path = Path(path)
    if not path.exists():
        raise ModelLoadError(
            f"{label} not found at {path}. "
            "Place the .pkl files in the models/ directory."
        )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        obj = joblib.load(path)
        # Deduplicate: log at most one version-mismatch warning per model load.
        if any("InconsistentVersionWarning" in str(warn.category) for warn in w):
            log.warning(
                f"{label}: sklearn version mismatch "
                f"(trained on an older version). Predictions may vary slightly."
            )
    log.info(f"Loaded {label} from {path}")
    return obj


@st.cache_resource(show_spinner=False)
def load_price_value_model():
    """
    Load the Random Forest Regressor that predicts value_score_calc.
    Target: quality_score / price_usd   (quality points per $1)
    Input: 12 features in MODEL_FEATURES order
    Returns: fitted sklearn RandomForestRegressor
    """
    return _safe_load(REGRESSOR_PKL, "Price-Value Regressor")


@st.cache_resource(show_spinner=False)
def load_price_tier_model():
    """
    Load the Random Forest Classifier that predicts price tier.
    Classes: ['AAA', 'Budget', 'Mid-range', 'Premium']
    Input: 12 features in MODEL_FEATURES order (Free games were excluded from training)
    Returns: fitted sklearn RandomForestClassifier
    """
    return _safe_load(CLASSIFIER_PKL, "Price Tier Classifier")


@st.cache_resource(show_spinner=False)
def load_scaler():
    """
    Load the StandardScaler fitted on the 12 model features.
    Returns: fitted sklearn StandardScaler
    Note: The saved RandomForest models do NOT require scaling at inference
    time—the scaler was used only for the LogisticRegression baseline in the
    notebook. The RF models use raw feature values directly.
    """
    return _safe_load(SCALER_PKL, "Feature Scaler")


def predict_value_score(game_profile: dict) -> float:
    """
    Predict value_score_calc for a game profile.

    Parameters
    ----------
    game_profile : dict with keys matching MODEL_FEATURES
        All missing keys default to 0.

    Returns
    -------
    float : predicted quality points per $1

    Notes
    -----
    The RF regressor does not require scaling at inference (the scaler was
    used only for the Logistic Regression baseline). We pass raw features.
    """
    from src.feature_engineering import build_model_input
    model = load_price_value_model()
    X = build_model_input(game_profile)
    return float(model.predict(X)[0])


def predict_price_tier(game_profile: dict) -> tuple[str, dict[str, float]]:
    """
    Predict the natural price tier for a game profile.

    Returns
    -------
    (tier_label, probabilities_dict)
        tier_label : str — e.g. 'Budget', 'Mid-range', 'Premium', 'AAA'
        probabilities_dict : {class_label: probability float}
    """
    from src.feature_engineering import build_model_input
    model = load_price_tier_model()
    X = build_model_input(game_profile)
    tier = str(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    proba_dict = {str(c): round(float(p), 4) for c, p in zip(model.classes_, proba)}
    return tier, proba_dict


def validate_model_features_match(df: pd.DataFrame) -> bool:
    """Return True if all MODEL_FEATURES columns are present in df."""
    return all(f in df.columns for f in MODEL_FEATURES)
