"""
test_day2.py -- Unit and integration tests for Day 2 features.

Focus:
  - Task 3: publisher_studio.py (Indie Developer Studio & What-If Simulator)
  - Radar chart rendering & market-normalized calculations
  - Profile building and feature contract preservation (MODEL_FEATURES, patforms_count)
  - ML inference integration (predict_value_score, predict_price_tier)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_data
from src.feature_engineering import apply_all_features, build_model_input
from src.config import MODEL_FEATURES, PRIMARY_GENRES
from src.model_loader import predict_value_score, predict_price_tier
from src.similarity import find_similar_games
from src.benchmarks import percentile_profile
from app.pages.publisher_studio import _render_dual_radar, PRESETS, TIER_COLORS, TIER_RANGES


@pytest.fixture(scope="module")
def dataset():
    df = load_data()
    df = apply_all_features(df)
    return df


class TestPublisherStudio:
    def test_presets_validity(self):
        """Ensure all presets have required fields and non-empty values."""
        for name, p in PRESETS.items():
            if p is None:
                continue
            assert "name" in p
            assert "genre" in p
            assert p["genre"] in PRIMARY_GENRES
            assert p["price"] >= 0
            assert 0 <= p["review_pct"] <= 100
            assert len(p["platforms"]) >= 1

    def test_model_input_from_preset(self, dataset):
        """Ensure preset translates into valid model inputs with all MODEL_FEATURES."""
        preset = PRESETS["Indie Action / Roguelike"]
        model_profile = {
            "quality_score":              float(preset["review_pct"]),
            "age_by_years":               float(preset["age_years"]),
            "categories_count":           float(preset["categories"]),
            "languages_count":            float(preset["languages"]),
            "peak_ccu":                   float(preset["peak_ccu"]),
            "log_reviews":                float(np.log1p(preset["total_rev"])),
            "genre_casual":               1.0 if preset["is_casual"] else 0.0,
            "genre_count":                float(preset["genre_count"]),
            "full_audio_languages_count": float(preset["audio_lang"]),
            "is_indie":                   1.0 if preset["is_indie"] else 0.0,
            "average_playtime_forever":   float(preset["playtime"]),
            "cat_single_player":          1.0 if preset["is_single"] else 0.0,
        }
        X = build_model_input(model_profile)
        assert list(X.columns) == MODEL_FEATURES
        assert not X.isna().any().any()

        val_score = predict_value_score(model_profile)
        assert isinstance(val_score, float)
        assert val_score > 0

        tier, proba = predict_price_tier(model_profile)
        assert tier in ["Budget", "Mid-range", "Premium", "AAA"]
        assert isinstance(proba, dict)
        assert abs(sum(proba.values()) - 1.0) < 0.01

    def test_patforms_count_preserved(self, dataset):
        """Preserve exact spelling patforms_count in game_values."""
        preset = PRESETS["Indie Action / Roguelike"]
        game_values = {
            "name":                     preset["name"],
            "primary_genre":            preset["genre"],
            "price":                    float(preset["price"]),
            "review_score_pct":         float(preset["review_pct"]) / 100.0,
            "owners_mid":               50000.0,
            "recommendations":          float(preset["total_rev"]),
            "peak_ccu":                 float(preset["peak_ccu"]),
            "average_playtime_forever": float(preset["playtime"]),
            "languages_count":          float(preset["languages"]),
            "patforms_count":           float(len(preset["platforms"])),
            "total_review":             float(preset["total_rev"]),
            "age_by_years":             float(preset["age_years"]),
        }
        assert "patforms_count" in game_values
        pct_df = percentile_profile(game_values, dataset, genre=preset["genre"])
        assert not pct_df.empty
        assert "patforms_count" in pct_df["metric"].values

    def test_render_radar_returns_figure(self, dataset):
        """Verify _render_dual_radar outputs a valid Plotly Figure without crashing."""
        preset = PRESETS["Cozy Narrative Adventure"]
        game_values = {
            "price":                    float(preset["price"]),
            "review_score_pct":         float(preset["review_pct"]) / 100.0,
            "recommendations":          float(preset["total_rev"]),
            "peak_ccu":                 float(preset["peak_ccu"]),
            "average_playtime_forever": float(preset["playtime"]),
            "languages_count":          float(preset["languages"]),
        }
        fig = _render_dual_radar(game_values, game_values, dataset, genre=preset["genre"])
        assert fig is not None
        assert len(fig.data) == 3  # Base + Alt + Genre trace

    def test_similar_games_integration(self, dataset):
        """Verify find_similar_games works for a simulated indie profile."""
        game_values = {
            "name":                     "Test Indie",
            "primary_genre":            "Action",
            "price":                    14.99,
            "review_score_pct":         0.80,
            "owners_mid":               20000.0,
            "recommendations":          300.0,
            "peak_ccu":                 500.0,
            "average_playtime_forever": 350.0,
            "languages_count":          5.0,
            "patforms_count":           2.0,
            "total_review":             300.0,
            "age_by_years":             1.0,
        }
        sim_df = find_similar_games(game_values, dataset, n=5)
        assert len(sim_df) == 5
        assert "similarity_score" in sim_df.columns
        assert (sim_df["similarity_score"] >= 0).all()
        assert (sim_df["similarity_score"] <= 1.01).all()


class TestGenreBenchmark:
    def test_genre_summary_returns_metrics(self, dataset):
        from src.analytics import genre_summary
        g_sum = genre_summary(dataset, min_games=20)
        assert not g_sum.empty
        assert "primary_genre" in g_sum.columns
        assert "median_price" in g_sum.columns
        assert "median_quality" in g_sum.columns
        assert "mean_platforms" in g_sum.columns

    def test_competition_genre_x_tier(self, dataset):
        from src.market_position import competition_genre_x_tier
        pivot = competition_genre_x_tier(dataset, min_genre_games=30)
        assert isinstance(pivot, pd.DataFrame)
        assert not pivot.empty
        assert "Budget" in pivot.columns
        assert "Mid-range" in pivot.columns

    def test_market_gap_signals(self, dataset):
        from src.market_position import compute_market_gap_signal
        gaps = compute_market_gap_signal(dataset, min_games=30)
        assert not gaps.empty
        assert "gap_signal_norm" in gaps.columns
        assert (gaps["gap_signal_norm"] >= 0).all()
        assert (gaps["gap_signal_norm"] <= 100.0).all()

    def test_release_density_by_genre_year(self, dataset):
        from src.market_position import release_density_by_genre_year
        pivot = release_density_by_genre_year(dataset, top_genres=5)
        assert isinstance(pivot, pd.DataFrame)
        assert not pivot.empty

    def test_genre_benchmark_render_import(self):
        from app.pages.genre_benchmark import render
        assert callable(render)


class TestGameComparison:
    def test_preset_showdowns_exist(self, dataset):
        from app.pages.game_comparison import PRESET_SHOWDOWNS
        for name, titles in PRESET_SHOWDOWNS.items():
            if not titles:
                continue
            found = dataset[dataset["name"].isin(titles)]
            assert len(found) >= 1

    def test_extract_model_profile(self, dataset):
        from app.pages.game_comparison import _extract_model_profile
        from src.config import MODEL_FEATURES
        from src.feature_engineering import build_model_input
        from src.model_loader import predict_value_score, predict_price_tier

        row = dataset.iloc[0]
        prof = _extract_model_profile(row)
        X = build_model_input(prof)
        assert list(X.columns) == MODEL_FEATURES

        val = predict_value_score(prof)
        assert isinstance(val, float)
        tier, _ = predict_price_tier(prof)
        assert tier in ["Budget", "Mid-range", "Premium", "AAA"]

    def test_multi_radar_generation(self, dataset):
        from app.pages.game_comparison import _render_multi_radar
        sample_rows = dataset.head(3).to_dict(orient="records")
        radar_cols = ["price", "review_score_pct", "total_review", "peak_ccu",
                      "average_playtime_forever", "languages_count", "patforms_count"]
        medians = dataset[radar_cols].median()
        fig = _render_multi_radar(sample_rows, medians)
        assert fig is not None
        assert len(fig.data) == 3

    def test_comparison_render_import(self):
        from app.pages.game_comparison import render
        assert callable(render)


class TestSimilarGamesPage:
    def test_popular_targets_exist(self, dataset):
        from app.pages.similar_games import POPULAR_TARGETS
        found = dataset[dataset["name"].isin(POPULAR_TARGETS)]
        assert len(found) >= 5

    def test_find_similar_games_returns_ranked(self, dataset):
        from src.similarity import find_similar_games
        sample_profile = {
            "name": "Target Query",
            "price": 19.99,
            "review_score_pct": 0.85,
            "owners_mid": 50000.0,
            "recommendations": 400.0,
            "peak_ccu": 600.0,
            "average_playtime_forever": 360.0,
            "languages_count": 6.0,
            "patforms_count": 2.0,
            "total_review": 400.0,
            "age_by_years": 1.0,
        }
        res = find_similar_games(sample_profile, dataset, n=8)
        assert len(res) == 8
        assert "similarity_score" in res.columns
        # Sorted descending
        scores = res["similarity_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_render_comparison_radar(self, dataset):
        from app.pages.similar_games import _render_comparison_radar
        target = dataset.iloc[0].to_dict()
        comps = dataset.iloc[1:4].to_dict(orient="records")
        radar_cols = ["price", "review_score_pct", "total_review", "peak_ccu",
                      "average_playtime_forever", "languages_count", "patforms_count"]
        medians = dataset[radar_cols].median()
        fig = _render_comparison_radar(target, comps, medians)
        assert fig is not None
        assert len(fig.data) == 4  # 1 target + 3 comps

    def test_similar_games_render_import(self):
        from app.pages.similar_games import render
        assert callable(render)


class TestAnomalyFinder:
    def test_compute_isolation_forest_anomalies(self, dataset):
        from app.pages.anomaly_finder import compute_isolation_forest_anomalies
        # Run on subset for fast testing
        sample = dataset.head(2000)
        anomalies = compute_isolation_forest_anomalies(sample, contamination=0.02)
        assert isinstance(anomalies, pd.DataFrame)
        assert not anomalies.empty
        assert "anomaly_score" in anomalies.columns
        assert (anomalies["anomaly_score"] >= 0).all()

    def test_hidden_gems_query(self, dataset):
        gems = dataset[
            (dataset["review_score_pct"] >= 0.90) &
            (dataset["total_review"] <= 3000) &
            (dataset["total_review"] >= 50) &
            (dataset["average_playtime_forever"] >= 120) &
            (dataset["price"] > 0)
        ]
        assert not gems.empty
        assert (gems["review_score_pct"] >= 0.90).all()

    def test_viral_breakouts_query(self, dataset):
        viral = dataset[
            (dataset["total_review"] >= 25000) |
            (dataset["peak_ccu"] >= 10000)
        ]
        assert not viral.empty
        assert len(viral) > 10

    def test_anomaly_render_import(self):
        from app.pages.anomaly_finder import render
        assert callable(render)




