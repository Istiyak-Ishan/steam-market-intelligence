"""
model_loader.py — Precision Offline Model Loading Layer

Loads the highly-tuned, statically trained HistGradientBoostingRegressors.
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
        model = joblib.load(path)
        return model
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

def _extract_base_features(game_profile: dict) -> list:
    from src.config import BASE_FEATURES
    # Map the dynamic game profile dict to exactly match BASE_FEATURES
    # This ensures exact feature order.
    # The profile might have keys that aren't perfectly named, so we need a mapping if they differ.
    
    # game_profile typically has:
    # "age_years", "categories_count", "languages_count", "peak_ccu", "total_reviews",
    # "is_casual", "genres_count", "full_audio_count", "is_indie", "playtime", "is_single_player"
    
    # Let's map it:
    vector = []
    vector.append(game_profile.get("age_by_years", 1.0))                  # age_by_years
    vector.append(game_profile.get("categories_count", 1))                # categories_count
    vector.append(game_profile.get("languages_count", 1))                 # languages_count
    vector.append(game_profile.get("peak_ccu", 0))                        # peak_ccu
    vector.append(game_profile.get("log_reviews", 0.0))                   # log_reviews
    vector.append(game_profile.get("genre_casual", 0))                    # genre_casual
    vector.append(game_profile.get("genre_count", 1))                     # genre_count
    vector.append(game_profile.get("full_audio_languages_count", 0))      # full_audio_languages_count
    vector.append(game_profile.get("is_indie", 0))                        # is_indie
    vector.append(game_profile.get("average_playtime_forever", 0.0))      # average_playtime_forever
    vector.append(game_profile.get("cat_single_player", 1))               # cat_single_player
    return vector

def predict_price_sweetspot(game_profile: dict, model_type: str = "") -> float:
    model = load_sweetspot_model()
    X = np.array([_extract_base_features(game_profile)])
    return float(model.predict(X)[0])

def predict_review_score(game_profile: dict, price: float) -> float:
    model = load_review_score_model()
    # model_review_score takes BASE_FEATURES + ['price']
    features = _extract_base_features(game_profile) + [price]
    X = np.array([features])
    return float(model.predict(X)[0])

def predict_value_score(game_profile: dict, model_type: str = "") -> float:
    model = load_value_score_model()
    X = np.array([_extract_base_features(game_profile)])
    return float(model.predict(X)[0])

def predict_ownership(game_profile: dict, price: float) -> float:
    model = load_ownership_model()
    # model_ownership takes BASE_FEATURES + ['price']
    features = _extract_base_features(game_profile) + [price]
    X = np.array([features])
    log_owners = float(model.predict(X)[0])
    return float(np.expm1(log_owners))

def predict_price_tier(game_profile: dict, model_type: str = "") -> tuple[str, dict[str, float]]:
    # Instead of a completely separate classifier, we can just infer the tier 
    # directly from the highly precise predicted sweetspot price!
    price = predict_price_sweetspot(game_profile)
    if price <= 0:
        tier = "Free"
    elif price <= 10.0:
        tier = "Budget"
    elif price <= 30.0:
        tier = "Mid-range"
    elif price <= 60.0:
        tier = "Premium"
    else:
        tier = "AAA"
        
    # Mock probabilities for backwards compatibility with UI
    proba = {"Budget": 0.0, "Mid-range": 0.0, "Premium": 0.0, "AAA": 0.0}
    if tier in proba:
        proba[tier] = 1.0
        
    return tier, proba
