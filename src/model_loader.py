"""
model_loader.py — Precision Offline Model Loading Layer

Loads the highly-tuned, statically trained ML models.
All dynamic on-the-fly fitting has been strictly prohibited.
"""
import joblib
import streamlit as st
import numpy as np
from pathlib import Path
from typing import Any
import logging

from src.config import MODELS_DIR

log = logging.getLogger(__name__)


class ModelLoadError(Exception):
    pass


def _safe_load(path: Path, label: str) -> Any:
    if not path.exists():
        raise ModelLoadError(f"{label} not found at {path}. Please run scripts/train_pipeline.py")
    try:
        return joblib.load(path)
    except Exception as e:
        raise ModelLoadError(f"Failed to load {label}: {e}")


@st.cache_resource(show_spinner=False)
def load_sweetspot_model():
    return _safe_load(MODELS_DIR / "model_sweetspot_price.pkl", "Sweetspot Regressor")


@st.cache_resource(show_spinner=False)
def load_review_score_model():
    return _safe_load(MODELS_DIR / "model_review_score.pkl", "Review Score Regressor")


@st.cache_resource(show_spinner=False)
def load_value_score_model():
    return _safe_load(MODELS_DIR / "model_value_score.pkl", "Value Score Regressor")


@st.cache_resource(show_spinner=False)
def load_ownership_model():
    return _safe_load(MODELS_DIR / "model_ownership.pkl", "Ownership Regressor")


@st.cache_resource(show_spinner=False)
def load_price_tier_clf():
    return _safe_load(MODELS_DIR / "model_price_tier_clf.pkl", "Price Tier Classifier")


@st.cache_resource(show_spinner=False)
def load_fair_price_clf():
    return _safe_load(MODELS_DIR / "model_fair_price_clf.pkl", "Fair Price Classifier")


def _base_feature_vector(game_profile: dict) -> list:
    """Build the BASE_FEATURES vector in correct order."""
    return [
        float(game_profile.get("age_by_years", 1.0)),
        float(game_profile.get("categories_count", 1)),
        float(game_profile.get("languages_count", 1)),
        float(game_profile.get("peak_ccu", 0)),
        float(game_profile.get("log_reviews", 0.0)),
        float(game_profile.get("genre_casual", 0)),
        float(game_profile.get("genre_count", 1)),
        float(game_profile.get("full_audio_languages_count", 0)),
        float(game_profile.get("is_indie", 0)),
        float(game_profile.get("average_playtime_forever", 0.0)),
        float(game_profile.get("cat_single_player", 1)),
    ]


# Keep old name for backward compatibility
_extract_base_features = _base_feature_vector


def predict_price_sweetspot(game_profile: dict, model_type: str = "") -> float:
    model = load_sweetspot_model()
    X = np.array([_base_feature_vector(game_profile)])
    return float(model.predict(X)[0])


def predict_review_score(game_profile: dict, price: float) -> float:
    model = load_review_score_model()
    features = _base_feature_vector(game_profile) + [float(price)]
    X = np.array([features])
    return float(model.predict(X)[0])


def predict_value_score(game_profile: dict, model_type: str = "") -> float:
    model = load_value_score_model()
    X = np.array([_base_feature_vector(game_profile)])
    return float(model.predict(X)[0])


def predict_ownership(game_profile: dict, price: float) -> float:
    model = load_ownership_model()
    features = _base_feature_vector(game_profile) + [float(price)]
    X = np.array([features])
    log_owners = float(model.predict(X)[0])
    return float(np.expm1(log_owners))


def predict_price_tier(game_profile: dict, price: float | None = None) -> tuple:
    """Predict price tier using the trained HistGradientBoostingClassifier."""
    clf = load_price_tier_clf()
    features = _base_feature_vector(game_profile)
    X = np.array([features])
    pred = int(clf.predict(X)[0])
    tier_names = ["Budget", "Mid-range", "Premium", "AAA"]
    tier = tier_names[pred] if 0 <= pred < len(tier_names) else "Budget"
    try:
        proba_arr = clf.predict_proba(X)[0]
        proba = {t: float(p) for t, p in zip(tier_names, proba_arr)}
    except AttributeError:
        proba = {t: (1.0 if t == tier else 0.0) for t in tier_names}
    return tier, proba


def predict_fair_price(game_profile: dict, price: float) -> tuple:
    """Predict whether a game is fairly priced. Returns (label, confidence)."""
    clf = load_fair_price_clf()
    features = _base_feature_vector(game_profile) + [float(price)]
    X = np.array([features])
    pred = int(clf.predict(X)[0])
    label = "Fair" if pred == 1 else "Overpriced"
    try:
        proba_arr = clf.predict_proba(X)[0]
        confidence = float(proba_arr[pred])
    except AttributeError:
        confidence = 1.0
    return label, confidence
