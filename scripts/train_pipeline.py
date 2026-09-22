import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from pathlib import Path

# Ensure paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "steam_games_cleaned.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Feature Engineering ──
def prep_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Preparing data...")
    df = df.copy()
    
    # Target 1: Price
    # Target 2: Quality Score (review_score_pct * 100)
    df['quality_score'] = df['review_score_pct'] * 100
    
    # Target 3: Value Score (quality / price) - avoid division by zero
    df['value_score_calc'] = np.where(df['price'] > 0, df['quality_score'] / df['price'], 0)
    
    # Target 4: log1p(highest_estimate_owner)
    df['log_owners'] = np.log1p(df['highest_estimate_owner'])
    
    # Features
    df['log_reviews'] = np.log1p(df['total_review'])
    df['genre_casual'] = df['genres'].fillna('').str.contains('Casual').astype(int)
    df['is_indie'] = df['genres'].fillna('').str.contains('Indie').astype(int)
    df['cat_single_player'] = df['categories'].fillna('').str.contains('Single-player').astype(int)
    
    # Count of full audio languages (comma separated string)
    df['full_audio_languages_count'] = df['full_audio_languages'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
    
    return df

def train_models():
    print(f"Loading data from {DATA_PATH}...")
    raw_df = pd.read_csv(DATA_PATH)
    df = prep_data(raw_df)
    
    # Common features for all models (exclude target leaks!)
    BASE_FEATURES = [
        "age_by_years",
        "categories_count",
        "languages_count",
        "peak_ccu",
        "log_reviews",
        "genre_casual",
        "genre_count",
        "full_audio_languages_count",
        "is_indie",
        "average_playtime_forever",
        "cat_single_player",
    ]
    
    # 1. Price Sweetspot Model
    # Filter: price > 0 and price <= 80 (remove free and crazy outliers)
    print("\n--- Training model_sweetspot_price ---")
    df_price = df[(df['price'] > 0) & (df['price'] <= 80)].copy()
    X_price = df_price[BASE_FEATURES]
    y_price = df_price['price']
    
    model_price = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=42)
    model_price.fit(X_price, y_price)
    score = model_price.score(X_price, y_price)
    print(f"Price Model R2: {score:.4f}")
    joblib.dump(model_price, MODELS_DIR / "model_sweetspot_price.pkl")
    
    # 2. Review Score (Quality) Model
    # Filter: games with reviews
    print("\n--- Training model_review_score ---")
    df_quality = df[df['has_reviews'] == 1].copy()
    X_quality = df_quality[BASE_FEATURES + ['price']]
    y_quality = df_quality['quality_score']
    
    model_quality = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, random_state=42)
    model_quality.fit(X_quality, y_quality)
    score = model_quality.score(X_quality, y_quality)
    print(f"Quality Model R2: {score:.4f}")
    joblib.dump(model_quality, MODELS_DIR / "model_review_score.pkl")
    
    # 3. Value Score Model
    print("\n--- Training model_value_score ---")
    df_value = df[(df['price'] > 0) & (df['price'] <= 80) & (df['has_reviews'] == 1)].copy()
    
    # Cap value score to 99th percentile to avoid crazy outliers where price is $0.99 and quality is 100
    cap = df_value['value_score_calc'].quantile(0.99)
    y_value = df_value['value_score_calc'].clip(upper=cap)
    
    X_value = df_value[BASE_FEATURES]
    model_value = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=42)
    model_value.fit(X_value, y_value)
    score = model_value.score(X_value, y_value)
    print(f"Value Model R2: {score:.4f}")
    joblib.dump(model_value, MODELS_DIR / "model_value_score.pkl")
    
    # 4. Ownership Model
    print("\n--- Training model_ownership ---")
    df_own = df[df['highest_estimate_owner'] > 0].copy()
    X_own = df_own[BASE_FEATURES + ['price']]
    y_own = df_own['log_owners']
    
    model_own = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=42)
    model_own.fit(X_own, y_own)
    score = model_own.score(X_own, y_own)
    print(f"Ownership Model R2: {score:.4f}")
    joblib.dump(model_own, MODELS_DIR / "model_ownership.pkl")
    
    print("\n✅ All 4 high-precision models trained and saved to models/ successfully!")

if __name__ == "__main__":
    train_models()
