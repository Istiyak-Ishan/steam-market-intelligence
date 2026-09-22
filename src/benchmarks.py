"""
benchmarks.py — Game Benchmark Engine for the Steam Market Intelligence Platform.

Compares a game profile against:
  1. The overall Steam market
  2. Its primary genre
  3. Its price tier (where applicable)

All benchmark values are calculated from the actual dataset — no hardcoded numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore


BENCHMARK_COLS = [
    "price", "review_score_pct", "owners_mid", "total_review",
    "peak_ccu", "average_playtime_forever", "patforms_count", "languages_count",
    "recommendations",
]

BENCHMARK_DISPLAY = {
    "price":                    "Price ($)",
    "review_score_pct":         "Review Score (%)",
    "owners_mid":               "Est. Owners",
    "total_review":             "Review Count",
    "peak_ccu":                 "Peak CCU",
    "average_playtime_forever": "Avg Playtime (min)",
    "patforms_count":           "Platform Count",
    "languages_count":          "Language Count",
    "recommendations":          "Recommendations",
}


def _percentile(series: pd.Series, value: float) -> float:
    """Return the percentile rank of value within series (0–100)."""
    clean = series.dropna().values
    if len(clean) == 0:
        return np.nan
    return float(percentileofscore(clean, value, kind="rank"))


def compute_market_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market-wide percentile lookup tables and medians.
    Returns a DataFrame with stats for every benchmark column.
    """
    rows = []
    for col in BENCHMARK_COLS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        rows.append({
            "metric":  col,
            "count":   len(series),
            "mean":    series.mean(),
            "median":  series.median(),
            "p25":     series.quantile(0.25),
            "p75":     series.quantile(0.75),
            "p90":     series.quantile(0.90),
            "p95":     series.quantile(0.95),
            "min":     series.min(),
            "max":     series.max(),
        })
    return pd.DataFrame(rows).set_index("metric")


def compute_genre_benchmarks(df: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    """
    Compute per-genre median benchmarks for all BENCHMARK_COLS.
    Returns a DataFrame indexed by primary_genre with median_{col} columns.
    """
    df = df[df["primary_genre"].notna()].copy()
    agg_dict = {}
    for col in BENCHMARK_COLS:
        if col in df.columns:
            agg_dict[f"median_{col}"] = (col, "median")
            agg_dict[f"mean_{col}"]   = (col, "mean")
            agg_dict[f"count_{col}"]  = (col, "count")
    agg_dict["game_count"] = ("app_id", "count")

    genre_stats = df.groupby("primary_genre").agg(**agg_dict).reset_index()
    return genre_stats[genre_stats["game_count"] >= min_games]


def percentile_profile(
    game_values: dict,
    df: pd.DataFrame,
    genre: str | None = None,
) -> pd.DataFrame:
    """
    Calculate the percentile rank of a game across all benchmark metrics.

    Parameters
    ----------
    game_values : dict mapping metric name → numeric value
    df          : full dataset (or genre-filtered subset)
    genre       : if provided, also compute genre-relative percentiles

    Returns
    -------
    DataFrame with columns: metric, value, market_pct, genre_pct (if genre given)
    """
    rows = []
    genre_df = df[df["primary_genre"] == genre] if genre else None

    for col in BENCHMARK_COLS:
        if col not in game_values or col not in df.columns:
            continue
        val = game_values[col]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        market_pct = _percentile(df[col], val)
        row = {
            "metric":      col,
            "display":     BENCHMARK_DISPLAY.get(col, col),
            "value":       val,
            "market_pct":  round(market_pct, 1),
        }
        if genre_df is not None and len(genre_df) >= 5:
            row["genre_pct"] = round(_percentile(genre_df[col], val), 1)
        rows.append(row)

    return pd.DataFrame(rows)


def benchmark_game(
    game_row: pd.Series | dict,
    df: pd.DataFrame,
    genre_benchmarks: pd.DataFrame | None = None,
) -> dict:
    """
    Produce a full benchmark report for a single game.

    Parameters
    ----------
    game_row         : Series or dict with game metrics
    df               : full dataset
    genre_benchmarks : pre-computed genre benchmark DataFrame (optional)

    Returns
    -------
    dict with keys:
        game_name, genre, price_tier,
        market_percentiles (DataFrame),
        genre_percentiles  (DataFrame),
        genre_medians      (dict),
    """
    if isinstance(game_row, pd.Series):
        game_row = game_row.to_dict()

    genre = game_row.get("primary_genre")
    game_values = {col: game_row.get(col) for col in BENCHMARK_COLS}

    # Market-wide percentiles
    market_pcts = percentile_profile(game_values, df, genre=genre)

    # Genre medians
    genre_medians = {}
    if genre and genre_benchmarks is not None:
        gm = genre_benchmarks[genre_benchmarks["primary_genre"] == genre]
        if not gm.empty:
            for col in BENCHMARK_COLS:
                key = f"median_{col}"
                if key in gm.columns:
                    genre_medians[col] = float(gm.iloc[0][key])

    return {
        "game_name":          game_row.get("name", "Unknown"),
        "genre":              genre,
        "price_tier":         game_row.get("price_tier"),
        "market_percentiles": market_pcts,
        "genre_medians":      genre_medians,
    }
