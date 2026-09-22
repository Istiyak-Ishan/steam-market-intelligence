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


def predict_value_score(game_profile: dict, model_type: str = "Random Forest (Pre-trained)") -> float:
    """
    Predict value_score_calc for a game profile.

    Parameters
    ----------
    game_profile : dict with keys matching MODEL_FEATURES
        All missing keys default to 0.
    model_type : str, default "Random Forest (Pre-trained)"

    Returns
    -------
    float : predicted quality points per $1

    Notes
    -----
    The RF regressor does not require scaling at inference (the scaler was
    used only for the Logistic Regression baseline). We pass raw features.
    """
    from src.feature_engineering import build_model_input
    
    if model_type == "Random Forest (Pre-trained)":
        model = load_price_value_model()
    else:
        model = load_dynamic_value_model(model_type)
        
    X = build_model_input(game_profile)
    
    if getattr(model, "is_scaled", False):
        scaler = load_scaler()
        X = scaler.transform(X)
        
    return float(model.predict(X)[0])


def predict_price_tier(game_profile: dict, model_type: str = "Random Forest (Pre-trained)") -> tuple[str, dict[str, float]]:
    """
    Predict the natural price tier for a game profile.

    Returns
    -------
    (tier_label, probabilities_dict)
        tier_label : str — e.g. 'Budget', 'Mid-range', 'Premium', 'AAA'
        probabilities_dict : {class_label: probability float}
    """
    from src.feature_engineering import build_model_input
    
    if model_type == "Random Forest (Pre-trained)":
        model = load_price_tier_model()
    else:
        model = load_dynamic_tier_model(model_type)
        
    X = build_model_input(game_profile)
    if getattr(model, "is_scaled", False):
        scaler = load_scaler()
        X = scaler.transform(X)
        
    tier = str(model.predict(X)[0])
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        proba_dict = {str(c): round(float(p), 4) for c, p in zip(model.classes_, proba)}
    else:
        proba_dict = {tier: 1.0}
        
    return tier, proba_dict


def validate_model_features_match(df: pd.DataFrame) -> bool:
    """Return True if all MODEL_FEATURES columns are present in df."""
    return all(f in df.columns for f in MODEL_FEATURES)


@st.cache_resource(show_spinner=False)
def load_dynamic_tier_model(model_type="Decision Tree"):
    from src.data_loader import load_data
    from src.feature_engineering import apply_all_features
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    
    df = load_data()
    df = apply_all_features(df)
    train_df = df[df["price"] > 0].dropna(subset=MODEL_FEATURES + ["price_tier"]).copy()
    X = train_df[MODEL_FEATURES]
    y = train_df["price_tier"]
    
    if model_type == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=10, random_state=42)
    elif model_type in ["Logistic Regression", "Linear/Logistic Regression"]:
        # Scale for Logistic Regression
        scaler = load_scaler()
        X_scaled = scaler.transform(X)
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        model.fit(X_scaled, y)
        model.is_scaled = True
        model.training_score = model.score(X_scaled, y)
        return model
    
    model.fit(X, y)
    model.is_scaled = False
    model.training_score = model.score(X, y)
    return model


@st.cache_resource(show_spinner=False)
def load_dynamic_value_model(model_type="Decision Tree"):
    from src.data_loader import load_data
    from src.feature_engineering import apply_all_features
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LinearRegression
    
    df = load_data()
    df = apply_all_features(df)
    train_df = df[df["price"] > 0].dropna(subset=MODEL_FEATURES + ["value_score_calc"]).copy()
    
    # Cap target outliers for regression stability
    target = train_df["value_score_calc"]
    cap = target.quantile(0.99)
    train_df["value_score_calc"] = target.clip(upper=cap)
    
    X = train_df[MODEL_FEATURES]
    y = train_df["value_score_calc"]
    
    if model_type == "Decision Tree":
        model = DecisionTreeRegressor(max_depth=10, random_state=42)
    elif model_type in ["Linear Regression", "Linear/Logistic Regression"]:
        scaler = load_scaler()
        X_scaled = scaler.transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)
        model.is_scaled = True
        model.training_score = model.score(X_scaled, y)
        return model
        
    model.fit(X, y)
    model.is_scaled = False
    model.training_score = model.score(X, y)
    return model


@st.cache_resource(show_spinner=False)
def load_dynamic_price_regressor(model_type="Random Forest (Pre-trained)"):
    """
    Train a continuous price regressor dynamically.
    Since we don't have a pre-trained price regressor pickle, 'Random Forest (Pre-trained)' 
    will also just train a dynamic RandomForestRegressor on the fly.
    """
    from src.data_loader import load_data
    from src.feature_engineering import apply_all_features
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    
    df = load_data()
    df = apply_all_features(df)
    
    # Filter valid prices and remove extreme outliers > $80
    train_df = df[(df["price"] > 0) & (df["price"] <= 80)].dropna(subset=MODEL_FEATURES + ["price"]).copy()
    
    X = train_df[MODEL_FEATURES]
    y = train_df["price"]
    
    if model_type == "Decision Tree":
        model = DecisionTreeRegressor(max_depth=10, random_state=42)
    elif model_type in ["Linear Regression", "Linear/Logistic Regression"]:
        scaler = load_scaler()
        X_scaled = scaler.transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)
        model.is_scaled = True
        model.training_score = model.score(X_scaled, y)
        return model
    else:
        # Fallback for Random Forest
        model = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42)
        
    model.fit(X, y)
    model.is_scaled = False
    model.training_score = model.score(X, y)
    return model


def predict_price_sweetspot(game_profile: dict, model_type: str = "Random Forest (Pre-trained)") -> float:
    """Predict the exact dollar sweetspot price for a game profile."""
    from src.feature_engineering import build_model_input
    
    model = load_dynamic_price_regressor(model_type)
    X = build_model_input(game_profile)
    
    if getattr(model, "is_scaled", False):
        scaler = load_scaler()
        X = scaler.transform(X)
        
    price_pred = float(model.predict(X)[0])
    # Cap negative or weird predictions
    return max(0.0, price_pred)
