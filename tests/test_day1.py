"""
tests/test_day1.py -- Day 1 Integration Test Suite
Steam Market Intelligence Platform

All assertions verified against the actual runtime API of each src/ module.
Run from project root:  python -m pytest tests/test_day1.py -v
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="module")
def df():
    from src.data_loader import load_data
    from src.feature_engineering import apply_all_features
    _df = load_data()
    return apply_all_features(_df)


@pytest.fixture(scope="module")
def sample_row(df):
    return df.dropna(subset=["primary_genre", "price", "review_score_pct"]).iloc[0]


@pytest.fixture(scope="module")
def model_features():
    return {
        "price": 19.99, "review_score_pct": 85.0, "owners_mid": 50000,
        "recommendations": 500, "peak_ccu": 1000, "average_playtime_forever": 200,
        "languages_count": 5, "patforms_count": 3, "age_by_years": 2.0,
    }


class TestDataLoader:
    def test_shape(self, df):
        assert df.shape[0] > 100000
        assert df.shape[1] > 40

    def test_key_columns(self, df):
        must_have = ["app_id", "name", "price", "review_score_pct", "owners_mid",
                     "primary_genre", "price_tier", "patforms_count", "age_by_years"]
        missing = [c for c in must_have if c not in df.columns]
        assert missing == [], f"Missing: {missing}"


class TestValidation:
    def test_return_structure(self, df):
        from src.validation import run_all_validations
        r = run_all_validations(df)
        assert "ok" in r and "errors" in r and "warnings" in r

    def test_ok_is_bool(self, df):
        from src.validation import run_all_validations
        assert isinstance(run_all_validations(df)["ok"], bool)

    def test_passes_on_clean_data(self, df):
        from src.validation import run_all_validations
        r = run_all_validations(df)
        assert r["ok"] is True, f"Errors: {r.get('errors')}"


class TestAnalytics:
    def test_dataset_summary_keys(self, df):
        from src.analytics import dataset_summary
        s = dataset_summary(df)
        for k in ["total_games", "unique_genres", "paid_games", "free_games",
                   "median_price", "mean_price", "median_review_pct"]:
            assert k in s, f"Missing key: {k}"

    def test_genre_summary_columns(self, df):
        from src.analytics import genre_summary
        g = genre_summary(df)
        for c in ["primary_genre", "game_count", "mean_price", "median_price", "mean_quality"]:
            assert c in g.columns, f"Missing col: {c}"
        assert len(g) > 0

    def test_price_tier_distribution(self, df):
        from src.analytics import price_tier_distribution
        t = price_tier_distribution(df)
        assert {"price_tier", "count", "pct"}.issubset(t.columns)

    def test_key_correlations(self, df):
        from src.analytics import key_correlations
        c = key_correlations(df)
        assert {"feature_a", "feature_b", "pearson", "spearman"}.issubset(c.columns)

    def test_price_distribution(self, df):
        from src.analytics import price_distribution
        p = price_distribution(df, max_price=80.0)
        assert len(p) > 0

    def test_release_volume_by_year(self, df):
        from src.analytics import release_volume_by_year
        rv = release_volume_by_year(df)
        assert isinstance(rv, pd.DataFrame) and len(rv) > 0


class TestBenchmarks:
    def test_return_keys(self, df, sample_row):
        from src.benchmarks import benchmark_game
        bm = benchmark_game(sample_row, df)
        for k in ["game_name", "genre", "price_tier", "market_percentiles", "genre_medians"]:
            assert k in bm, f"Missing key: {k}"

    def test_market_percentiles_is_dict(self, df, sample_row):
        from src.benchmarks import benchmark_game
        assert isinstance(benchmark_game(sample_row, df)["market_percentiles"], (dict, pd.DataFrame))

    def test_genre_medians_is_dict(self, df, sample_row):
        from src.benchmarks import benchmark_game
        assert isinstance(benchmark_game(sample_row, df)["genre_medians"], dict)


class TestMarketPosition:
    def test_new_columns_added(self, df):
        from src.market_position import compute_market_positions
        mdf = compute_market_positions(df.head(500))
        for c in ["price_above_median", "quality_above_median", "owners_above_median",
                   "price_quality_pos", "quality_owners_pos", "price_owners_pos"]:
            assert c in mdf.columns, f"Missing column: {c}"

    def test_shape_preserved(self, df):
        from src.market_position import compute_market_positions
        s = df.head(500)
        assert compute_market_positions(s).shape[0] == s.shape[0]


class TestSimilarity:
    def test_returns_dataframe(self, df, sample_row):
        from src.similarity import find_similar_games
        assert isinstance(find_similar_games(sample_row, df, n=5), pd.DataFrame)

    def test_similarity_score_column(self, df, sample_row):
        from src.similarity import find_similar_games
        assert "similarity_score" in find_similar_games(sample_row, df, n=5).columns

    def test_n_rows_returned(self, df, sample_row):
        from src.similarity import find_similar_games
        assert len(find_similar_games(sample_row, df, n=5)) == 5

    def test_display_cols_present(self, df, sample_row):
        from src.similarity import find_similar_games, DISPLAY_COLS
        result = find_similar_games(sample_row, df, n=3)
        for c in DISPLAY_COLS:
            if c in df.columns:
                assert c in result.columns, f"Missing DISPLAY_COL: {c}"

    def test_scores_between_0_and_1(self, df, sample_row):
        from src.similarity import find_similar_games
        r = find_similar_games(sample_row, df, n=5)
        assert (r["similarity_score"] >= 0).all() and (r["similarity_score"] <= 1).all()


class TestSegmentation:
    def test_run_segmentation_returns_dataframe(self, df):
        from src.segmentation import run_segmentation
        assert isinstance(run_segmentation(df, force=False), pd.DataFrame)

    def test_cluster_label_column(self, df):
        from src.segmentation import run_segmentation
        seg = run_segmentation(df, force=False)
        assert "cluster_label" in seg.columns or "cluster" in seg.columns, (
            f"Segmentation cols: {list(seg.columns)}"
        )

    def test_no_nan_in_cluster(self, df):
        from src.segmentation import run_segmentation
        seg = run_segmentation(df, force=False)
        col = "cluster_label" if "cluster_label" in seg.columns else "cluster"
        assert seg[col].notna().all()


class TestModelLoader:
    def test_predict_value_score_returns_float(self, model_features):
        from src.model_loader import predict_value_score
        val = predict_value_score(model_features)
        assert isinstance(val, float) and val >= 0

    def test_predict_price_tier_returns_tuple(self, model_features):
        from src.model_loader import predict_price_tier
        tier, proba = predict_price_tier(model_features)
        assert isinstance(tier, str) and isinstance(proba, dict) and len(proba) > 0

    def test_predict_price_tier_label_valid(self, model_features):
        from src.model_loader import predict_price_tier
        tier, _ = predict_price_tier(model_features)
        assert tier in {"AAA", "Budget", "Mid-range", "Premium", "Free"}, f"Bad tier: {tier}"

    def test_validate_model_features_match(self, df):
        from src.model_loader import validate_model_features_match
        assert validate_model_features_match(df) is True

    def test_models_load_without_error(self):
        from src.model_loader import load_price_value_model, load_price_tier_model
        assert load_price_value_model() is not None
        assert load_price_tier_model() is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

