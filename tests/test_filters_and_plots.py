import sys
from pathlib import Path
ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from src.data_loader import load_data
from app.pages.market_explorer import (
    _apply_filters,
    _genre_price_boxplot,
    _price_distribution,
    _price_vs_quality,
    _ownership_bubble,
    _genre_value_ranking,
    _release_timeline,
    _genre_price_heatmap,
    _cohort_heatmap,
    TIER_ORDER,
    NON_GAME_OR_DESCRIPTOR_GENRES
)
import src.analytics as an

df = load_data()
print(f"Dataset loaded: {len(df)} rows")

# Test suite of filter combinations
test_filters = [
    {
        "name": "Default (unfiltered)",
        "genres": [], "price_range": (0.0, 100.0), "year_range": (2005, 2026),
        "review_min": 0, "platforms": [], "lang_min": 1, "owner_min": 0, "tiers": []
    },
    {
        "name": "Free Only",
        "genres": [], "price_range": (0.0, 0.0), "year_range": (2005, 2026),
        "review_min": 0, "platforms": [], "lang_min": 1, "owner_min": 0, "tiers": ["Free"]
    },
    {
        "name": "AAA Only",
        "genres": [], "price_range": (50.0, 100.0), "year_range": (2005, 2026),
        "review_min": 0, "platforms": [], "lang_min": 1, "owner_min": 0, "tiers": ["AAA"]
    },
    {
        "name": "Single Small Genre (Racing)",
        "genres": ["Racing"], "price_range": (0.0, 100.0), "year_range": (2005, 2026),
        "review_min": 0, "platforms": [], "lang_min": 1, "owner_min": 0, "tiers": []
    },
    {
        "name": "Linux + High Review (>=80%) + 10+ Langs",
        "genres": [], "price_range": (0.0, 100.0), "year_range": (2015, 2025),
        "review_min": 80, "platforms": ["Linux"], "lang_min": 10, "owner_min": 50000, "tiers": []
    },
    {
        "name": "High Ownership (1M+)",
        "genres": [], "price_range": (0.0, 100.0), "year_range": (2005, 2026),
        "review_min": 0, "platforms": [], "lang_min": 1, "owner_min": 1000000, "tiers": []
    },
    {
        "name": "Narrow Year (2025-2026)",
        "genres": [], "price_range": (0.0, 100.0), "year_range": (2025, 2026),
        "review_min": 0, "platforms": [], "lang_min": 1, "owner_min": 0, "tiers": []
    },
    {
        "name": "Borderline sample size (~15-30 games)",
        "genres": ["Massively Multiplayer"], "price_range": (20.0, 50.0), "year_range": (2020, 2025),
        "review_min": 70, "platforms": ["Mac"], "lang_min": 5, "owner_min": 0, "tiers": []
    }
]

for tf in test_filters:
    print(f"\n--- Testing: {tf['name']} ---")
    fdf = _apply_filters(
        df,
        genres=tf["genres"],
        price_range=tf["price_range"],
        year_range=tf["year_range"],
        review_min=tf["review_min"],
        platforms=tf["platforms"],
        lang_min=tf["lang_min"],
        owner_min=tf["owner_min"],
        tiers=tf["tiers"]
    )
    print(f"Retained rows: {len(fdf)}")
    if len(fdf) < 10:
        print("Skipping plots (fdf < 10, as UI does)")
        continue

    # Test all plots and options
    try:
        fig1 = _genre_price_boxplot(fdf)
        fig2 = _price_distribution(fdf)
        fig3_1 = _price_vs_quality(fdf, scale_mode="Focused ($0–$70 Linear)", show_trend=True)
        fig3_2 = _price_vs_quality(fdf, scale_mode="Log Scale (log₁₀ Price)", show_trend=True)
        fig3_3 = _price_vs_quality(fdf, scale_mode="Full Catalog (Uncapped)", show_trend=False)
        fig4_1 = _ownership_bubble(fdf, scale_mode="Focused ($0–$70 Linear)", color_by="Top Genres (Clean Palette)", y_metric="Estimated Owners (with Jitter)")
        fig4_2 = _ownership_bubble(fdf, scale_mode="Log Scale (log₁₀ Price)", color_by="Price Tier", y_metric="Total Review Count (Continuous log₁₀)")
        fig5_1 = _genre_value_ranking(fdf, metric_choice="Log Cost-Efficiency (Review% / log₂(Price + 1))", include_software=False)
        fig5_2 = _genre_value_ranking(fdf, metric_choice="Playtime Hours per Dollar (Longevity)", include_software=True)
        fig5_3 = _genre_value_ranking(fdf, metric_choice="Quality Points per $1 (Price ≥ $4.99 Floor)", include_software=False)
        fig6_1 = _release_timeline(fdf, view_mode="Annual Volume (Multi-Line)", include_2026=False, y_scale="Linear")
        fig6_2 = _release_timeline(fdf, view_mode="Market Share % (100% Normalized Area)", include_2026=True, y_scale="Linear")
        fig6_3 = _release_timeline(fdf, view_mode="Stacked Area (Total Volume)", include_2026=False, y_scale="Log Scale (log₁₀)")
        fig7 = _genre_price_heatmap(fdf)
        fig8_1 = _cohort_heatmap(fdf, heat_mode="Row-Normalized (% of Year's Releases)", include_2026=False)
        fig8_2 = _cohort_heatmap(fdf, heat_mode="Log-Scaled Volume (log₁₀ Count)", include_2026=True)
        fig8_3 = _cohort_heatmap(fdf, heat_mode="Absolute Game Count", include_2026=False)

        # Tab 3 tables
        clean_fdf = fdf[~fdf["primary_genre"].isin(NON_GAME_OR_DESCRIPTOR_GENRES) & fdf["primary_genre"].notna()]
        g_counts = clean_fdf["primary_genre"].value_counts()
        valid_g = g_counts[g_counts >= 15].index
        g_own = (
            clean_fdf[clean_fdf["primary_genre"].isin(valid_g)]
            .groupby("primary_genre")
            .agg(
                games=("app_id", "count"),
                avg_owners=("owners_mid", "mean"),
                p75_owners=("owners_mid", lambda x: np.percentile(x.dropna(), 75) if len(x.dropna()) > 0 else 0),
                breakout_pct=("owners_mid", lambda x: (x > 20000).mean() * 100),
                median_price=("price", "median"),
                median_review=("review_score_pct", lambda x: x.dropna().median() * 100 if len(x.dropna()) > 0 else np.nan),
            )
            .reset_index()
            .sort_values("avg_owners", ascending=False)
        )
        top_raw = an.top_games_by_owners(fdf, n=15)
        print("All plots and tables generated successfully!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
