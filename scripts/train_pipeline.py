"""
train_pipeline.py -- Offline ML Training Pipeline for Steam Market Intelligence.

Trains 4 precision models with proper train/test splits and evaluation metrics.
Outputs model artifacts (.pkl) and a training report.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix,
)
from sklearn.inspection import permutation_importance
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "steam_games_cleaned.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def prep_data(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for training."""
    print("Preparing data...")
    df = df.copy()

    # Cap outliers at 1st-99th percentile
    for col in ["price", "peak_ccu"]:
        p01, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lower=p01, upper=p99)
        print(f"  Capped {col}: [{p01:.2f}, {p99:.2f}]")

    # Targets
    df["quality_score"] = df["review_score_pct"] * 100
    df["value_score_calc"] = np.where(df["price"] > 0, df["quality_score"] / df["price"], 0)
    df["log_owners"] = np.log1p(df["highest_estimate_owner"])

    # Features
    df["log_reviews"] = np.log1p(df["total_review"])
    df["genre_casual"] = df["genres"].fillna("").str.contains("Casual").astype(int)
    df["is_indie"] = df["genres"].fillna("").str.contains("Indie").astype(int)
    df["cat_single_player"] = df["categories"].fillna("").str.contains("Single-player").astype(int)
    df["full_audio_languages_count"] = (
        df["full_audio_languages"].fillna("").apply(lambda x: len(x.split(",")) if x else 0)
    )

    return df


BASE_FEATURES = [
    "age_by_years", "categories_count", "languages_count", "peak_ccu",
    "log_reviews", "genre_casual", "genre_count", "full_audio_languages_count",
    "is_indie", "average_playtime_forever", "cat_single_player",
]


def train_models():
    print(f"Loading data from {DATA_PATH}...")
    raw_df = pd.read_csv(DATA_PATH)
    df = prep_data(raw_df)

    report_lines = ["# Model Training Report\n"]

    # =====================================================================
    # 1. Price Sweetspot Regressor
    # =====================================================================
    print("\n--- Training model_sweetspot_price ---")
    df_price = df[(df["price"] > 0) & (df["price"] <= 80)].dropna(subset=BASE_FEATURES).copy()
    X = df_price[BASE_FEATURES]
    y = df_price["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"  Test R2={r2:.4f}, RMSE=${rmse:.2f}, MAE=${mae:.2f}")
    joblib.dump(model, MODELS_DIR / "model_sweetspot_price.pkl")
    report_lines.append(f"## 1. Price Sweetspot Regressor\n- Test R2: {r2:.4f}\n- Test RMSE: ${rmse:.2f}\n- Test MAE: ${mae:.2f}\n- Train size: {len(X_train):,} | Test size: {len(X_test):,}\n")

    # =====================================================================
    # 2. Review Score (Quality) Regressor
    # =====================================================================
    print("\n--- Training model_review_score ---")
    df_quality = df[df["has_reviews"] == 1].dropna(subset=BASE_FEATURES + ["price"]).copy()
    X = df_quality[BASE_FEATURES + ["price"]]
    y = df_quality["quality_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"  Test R2={r2:.4f}, RMSE={rmse:.2f}%, MAE={mae:.2f}%")
    joblib.dump(model, MODELS_DIR / "model_review_score.pkl")
    report_lines.append(f"## 2. Review Score Regressor\n- Test R2: {r2:.4f}\n- Test RMSE: {rmse:.2f}%\n- Test MAE: {mae:.2f}%\n- Train size: {len(X_train):,} | Test size: {len(X_test):,}\n")

    # =====================================================================
    # 3. Value Score Regressor
    # =====================================================================
    print("\n--- Training model_value_score ---")
    df_value = df[(df["price"] > 0) & (df["price"] <= 80) & (df["has_reviews"] == 1)].dropna(subset=BASE_FEATURES).copy()
    cap = df_value["value_score_calc"].quantile(0.99)
    df_value["value_score_calc"] = df_value["value_score_calc"].clip(upper=cap)
    X = df_value[BASE_FEATURES]
    y = df_value["value_score_calc"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"  Test R2={r2:.4f}, RMSE={rmse:.2f}")
    joblib.dump(model, MODELS_DIR / "model_value_score.pkl")
    report_lines.append(f"## 3. Value Score Regressor\n- Test R2: {r2:.4f}\n- Test RMSE: {rmse:.2f}\n- Train size: {len(X_train):,} | Test size: {len(X_test):,}\n")

    # =====================================================================
    # 4. Ownership Regressor
    # =====================================================================
    print("\n--- Training model_ownership ---")
    df_own = df[df["highest_estimate_owner"] > 0].dropna(subset=BASE_FEATURES + ["price"]).copy()
    X = df_own[BASE_FEATURES + ["price"]]
    y = df_own["log_owners"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"  Test R2={r2:.4f}, RMSE={rmse:.4f}")
    joblib.dump(model, MODELS_DIR / "model_ownership.pkl")
    report_lines.append(f"## 4. Ownership Regressor\n- Test R2: {r2:.4f}\n- Test RMSE: {rmse:.4f}\n- Train size: {len(X_train):,} | Test size: {len(X_test):,}\n")

    # Compute and save permutation importances for ownership model (strongest model)
    print("\n--- Computing permutation importance (ownership model) ---")
    perm_result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(MODELS_DIR / "permutation_importance.csv", index=False)
    print(perm_df.to_string(index=False))

    # =====================================================================
    # 5. Price Tier Classifier (Real trained classifier)
    # =====================================================================
    print("\n--- Training model_price_tier_classifier ---")
    df_clf = df[(df["price"] > 0) & (df["has_reviews"] == 1)].dropna(subset=BASE_FEATURES).copy()
    # Create target from price_tier column
    tier_map = {"Budget": 0, "Mid-range": 1, "Premium": 2, "AAA": 3}
    df_clf = df_clf[df_clf["price_tier"].isin(tier_map.keys())].copy()
    df_clf["tier_label"] = df_clf["price_tier"].map(tier_map)

    X = df_clf[BASE_FEATURES]
    y = df_clf["tier_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    tier_names = ["Budget", "Mid-range", "Premium", "AAA"]
    cr = classification_report(y_test, y_pred, target_names=tier_names)
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Test Accuracy: {acc:.4f}")
    print(cr)
    joblib.dump(clf, MODELS_DIR / "model_price_tier_clf.pkl")

    # Save confusion matrix and classification report for the UI
    cm_df = pd.DataFrame(cm, index=tier_names, columns=tier_names)
    cm_df.to_csv(MODELS_DIR / "confusion_matrix.csv")
    with open(MODELS_DIR / "classification_report.txt", "w") as f:
        f.write(cr)

    report_lines.append(f"## 5. Price Tier Classifier\n- Test Accuracy: {acc:.4f}\n- Train size: {len(X_train):,} | Test size: {len(X_test):,}\n\n### Classification Report\n```\n{cr}\n```\n\n### Confusion Matrix\n```\n{cm_df.to_string()}\n```\n")

    # =====================================================================
    # 6. Fair Price Classifier (Model 4 per proposal)
    # Fair = (price <= genre median) OR (value_score >= genre median value_score)
    # =====================================================================
    print("\n--- Training model_fair_price_classifier ---")
    df_fair = df[(df["price"] > 0) & (df["has_reviews"] == 1)].dropna(
        subset=BASE_FEATURES + ["price", "value_score_calc", "primary_genre"]
    ).copy()

    genre_med_price = df_fair.groupby("primary_genre")["price"].transform("median")
    genre_med_value = df_fair.groupby("primary_genre")["value_score_calc"].transform("median")
    df_fair["is_fair"] = (
        (df_fair["price"] <= genre_med_price) | (df_fair["value_score_calc"] >= genre_med_value)
    ).astype(int)

    X_fair = df_fair[BASE_FEATURES + ["price"]]
    y_fair = df_fair["is_fair"]
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X_fair, y_fair, test_size=0.2, random_state=42, stratify=y_fair
    )

    from sklearn.tree import DecisionTreeClassifier
    fair_clf = DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=42)
    fair_clf.fit(X_train_f, y_train_f)
    y_pred_f = fair_clf.predict(X_test_f)
    acc_f = accuracy_score(y_test_f, y_pred_f)
    cr_f = classification_report(y_test_f, y_pred_f, target_names=["Overpriced", "Fair"])
    print(f"  Test Accuracy: {acc_f:.4f}")
    print(cr_f)
    joblib.dump(fair_clf, MODELS_DIR / "model_fair_price_clf.pkl")

    fi_df = pd.DataFrame({
        "feature": list(X_fair.columns),
        "importance": fair_clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(MODELS_DIR / "fair_price_feature_importance.csv", index=False)
    with open(MODELS_DIR / "fair_price_classification_report.txt", "w") as f:
        f.write(cr_f)

    report_lines.append(f"## 6. Fair Price Classifier\n- Test Accuracy: {acc_f:.4f}\n- Definition: Fair = (price <= genre median) OR (value_score >= genre median)\n- Algorithm: DecisionTree(max_depth=8)\n- Train size: {len(X_train_f):,} | Test size: {len(X_test_f):,}\n\n### Classification Report\n```\n{cr_f}\n```\n")

    # Save full report
    report_path = MODELS_DIR / "training_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nTraining report saved to {report_path}")
    print("\nAll 6 models trained and saved successfully!")


if __name__ == "__main__":
    train_models()
