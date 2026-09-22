"""
market_position.py — Market positioning and competition density engines.

Provides:
  1. Market Position Engine (price × quality quadrants)
  2. Competition Density Engine (segment crowding analysis)
  3. Market Gap Signal (transparent analytical indicator)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════════════
# ── MARKET POSITION ENGINE ─────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def compute_market_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add descriptive market-position columns based on percentile thresholds.

    Position is relative to the full market (median split).
    Labels are descriptive — no value judgements ('good', 'bad', 'winner').

    Columns added:
        price_above_median    : bool
        quality_above_median  : bool
        owners_above_median   : bool
        price_quality_pos     : str  (4 descriptive categories)
        quality_owners_pos    : str  (4 descriptive categories)
        price_owners_pos      : str  (4 descriptive categories)
    """
    df = df.copy()

    price_med   = df["price"].median()
    quality_med = df["review_score_pct"].dropna().median()
    owners_med  = df["owners_mid"].median()

    df["price_above_median"]   = df["price"] >= price_med
    df["quality_above_median"] = df["review_score_pct"] >= quality_med
    df["owners_above_median"]  = df["owners_mid"] >= owners_med

    # Price × Quality
    def _pq_pos(row):
        if pd.isna(row.get("review_score_pct")):
            return "No Review Data"
        hi_p = row["price_above_median"]
        hi_q = row["quality_above_median"]
        if     hi_p and     hi_q:  return "Higher-Price / Higher-Quality"
        if     hi_p and not hi_q:  return "Higher-Price / Lower-Quality"
        if not hi_p and     hi_q:  return "Lower-Price / Higher-Quality"
        return                            "Lower-Price / Lower-Quality"

    # Quality × Owners
    def _qo_pos(row):
        if pd.isna(row.get("review_score_pct")):
            return "No Review Data"
        hi_q = row["quality_above_median"]
        hi_o = row["owners_above_median"]
        if     hi_q and     hi_o:  return "Higher-Quality / Higher-Ownership"
        if     hi_q and not hi_o:  return "Higher-Quality / Lower-Ownership"
        if not hi_q and     hi_o:  return "Lower-Quality / Higher-Ownership"
        return                            "Lower-Quality / Lower-Ownership"

    # Price × Owners
    def _po_pos(row):
        hi_p = row["price_above_median"]
        hi_o = row["owners_above_median"]
        if     hi_p and     hi_o:  return "Higher-Price / Higher-Ownership"
        if     hi_p and not hi_o:  return "Higher-Price / Lower-Ownership"
        if not hi_p and     hi_o:  return "Lower-Price / Higher-Ownership"
        return                            "Lower-Price / Lower-Ownership"

    df["price_quality_pos"]  = df.apply(_pq_pos, axis=1)
    df["quality_owners_pos"] = df.apply(_qo_pos, axis=1)
    df["price_owners_pos"]   = df.apply(_po_pos, axis=1)

    return df


def position_distribution(df: pd.DataFrame, position_col: str) -> pd.DataFrame:
    """Return game count per market position category."""
    counts = df[position_col].value_counts().reset_index()
    counts.columns = ["position", "count"]
    total = counts["count"].sum()
    counts["pct"] = (counts["count"] / total * 100).round(2)
    return counts


def game_market_position(game_values: dict, df: pd.DataFrame) -> dict:
    """
    Describe a single game's market position relative to the full dataset.

    Parameters
    ----------
    game_values : dict with keys: price, review_score_pct, owners_mid
    df          : full dataset

    Returns
    -------
    dict with position labels and threshold values
    """
    price_med   = float(df["price"].median())
    quality_med = float(df["review_score_pct"].dropna().median())
    owners_med  = float(df["owners_mid"].median())

    price   = game_values.get("price", np.nan)
    quality = game_values.get("review_score_pct", np.nan)
    owners  = game_values.get("owners_mid", np.nan)

    def _label(val, med, hi_lbl, lo_lbl):
        if pd.isna(val):
            return "Unknown"
        return hi_lbl if val >= med else lo_lbl

    return {
        "price":              price,
        "quality":            quality,
        "owners":             owners,
        "price_threshold":    price_med,
        "quality_threshold":  quality_med * 100,
        "owners_threshold":   owners_med,
        "price_position":     _label(price,   price_med,   "Above Median Price", "Below Median Price"),
        "quality_position":   _label(quality, quality_med, "Above Median Quality", "Below Median Quality"),
        "owners_position":    _label(owners,  owners_med,  "Above Median Ownership", "Below Median Ownership"),
        "price_quality_pos":  _pq_label(price, price_med, quality, quality_med),
    }


def _pq_label(price, price_med, quality, quality_med) -> str:
    if pd.isna(quality):
        return "No Review Data"
    hi_p = price >= price_med if not pd.isna(price) else False
    hi_q = quality >= quality_med if not pd.isna(quality) else False
    if     hi_p and     hi_q: return "Higher-Price / Higher-Quality"
    if     hi_p and not hi_q: return "Higher-Price / Lower-Quality"
    if not hi_p and     hi_q: return "Lower-Price / Higher-Quality"
    return                           "Lower-Price / Lower-Quality"


# ════════════════════════════════════════════════════════════════════════════════
# ── COMPETITION DENSITY ENGINE ─────────────────────────────────────════════════
# ════════════════════════════════════════════════════════════════════════════════

def competition_by_genre(df: pd.DataFrame, min_games: int = 1) -> pd.DataFrame:
    """Game count by primary genre, sorted descending."""
    counts = (
        df["primary_genre"]
        .dropna()
        .value_counts()
        .reset_index()
    )
    counts.columns = ["genre", "game_count"]
    return counts[counts["game_count"] >= min_games].reset_index(drop=True)


def competition_by_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Game count by price tier."""
    tier_order = ["Free", "Budget", "Mid-range", "Premium", "AAA"]
    counts = df["price_tier"].value_counts().reindex(tier_order, fill_value=0).reset_index()
    counts.columns = ["price_tier", "game_count"]
    return counts


def competition_genre_x_tier(df: pd.DataFrame, min_genre_games: int = 50) -> pd.DataFrame:
    """
    Game count cross-tabulated by primary_genre × price_tier.
    Returns a pivot table suitable for a heatmap.
    """
    tier_order = ["Free", "Budget", "Mid-range", "Premium", "AAA"]
    df = df[df["primary_genre"].notna()].copy()
    genre_counts = df["primary_genre"].value_counts()
    valid_genres = genre_counts[genre_counts >= min_genre_games].index

    sub = df[df["primary_genre"].isin(valid_genres)]
    pivot = (
        sub.groupby(["primary_genre", "price_tier"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[t for t in tier_order if t in sub["price_tier"].unique()], fill_value=0)
    )
    return pivot


def competition_by_quality_band(df: pd.DataFrame, min_games: int = 50) -> pd.DataFrame:
    """
    Game count by genre × quality band (≥80%, 60–80%, <60% positive).
    Returns a pivot table.
    """
    df = df[df["review_score_pct"].notna() & df["primary_genre"].notna()].copy()

    df["quality_band"] = pd.cut(
        df["review_score_pct"],
        bins=[-0.01, 0.60, 0.80, 1.01],
        labels=["<60% Positive", "60–80% Positive", "≥80% Positive"],
    )

    genre_counts = df["primary_genre"].value_counts()
    valid_genres = genre_counts[genre_counts >= min_games].index
    sub = df[df["primary_genre"].isin(valid_genres)]

    pivot = (
        sub.groupby(["primary_genre", "quality_band"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    return pivot


def release_density_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Game release count per year (reuses analytics.release_volume_by_year)."""
    from src.analytics import release_volume_by_year
    return release_volume_by_year(df)


def release_density_by_genre_year(
    df: pd.DataFrame,
    top_genres: int = 8,
    min_games: int = 30,
) -> pd.DataFrame:
    """
    Release count per primary_genre × year for the top N genres.
    Returns a pivot table (genres as columns, years as index).
    """
    if "release_year" not in df.columns:
        from src.feature_engineering import add_release_year
        df = add_release_year(df)

    df = df[df["primary_genre"].notna() & df["release_year"].notna()].copy()
    top_genre_list = (
        df["primary_genre"].value_counts().head(top_genres).index.tolist()
    )
    sub = df[df["primary_genre"].isin(top_genre_list)]
    pivot = (
        sub.groupby(["release_year", "primary_genre"])
        .size()
        .unstack(fill_value=0)
    )
    pivot.index = pivot.index.astype(int)
    return pivot


# ════════════════════════════════════════════════════════════════════════════════
# ── MARKET GAP SIGNAL ──────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def compute_market_gap_signal(
    df: pd.DataFrame,
    min_games: int = 50,
) -> pd.DataFrame:
    """
    Compute a transparent Market Gap Signal per genre × price_tier segment.

    Methodology (clearly documented):
      - demand_indicator   = median(owners_mid) × median(review_score_pct)
                             (high ownership + high quality → observed demand)
      - supply_indicator   = game_count
                             (number of competing games in segment)
      - gap_signal         = demand_indicator / (1 + supply_indicator)
                             (higher = less supply relative to observed demand)

    This is a descriptive analytical indicator based on dataset characteristics.
    It does NOT guarantee commercial opportunity or profitability.

    Returns
    -------
    DataFrame with columns:
        genre, price_tier, game_count,
        median_owners, median_quality,
        demand_indicator, supply_indicator, gap_signal
    Sorted by gap_signal descending.
    """
    tier_order = ["Budget", "Mid-range", "Premium", "AAA"]
    df = df[
        df["primary_genre"].notna() &
        df["price_tier"].isin(tier_order)
    ].copy()

    genre_counts = df["primary_genre"].value_counts()
    valid_genres = genre_counts[genre_counts >= min_games].index
    df = df[df["primary_genre"].isin(valid_genres)]

    grp = df.groupby(["primary_genre", "price_tier"]).agg(
        game_count=("app_id", "count"),
        median_owners=("owners_mid", "median"),
        median_quality=("review_score_pct", lambda x: x.dropna().median() * 100),
    ).reset_index()

    grp["demand_indicator"] = (
        grp["median_owners"] * grp["median_quality"].fillna(0) / 100
    )
    grp["supply_indicator"] = grp["game_count"]
    grp["gap_signal"] = grp["demand_indicator"] / (1 + grp["supply_indicator"])

    # Normalise gap_signal to 0–100 for interpretability
    max_sig = grp["gap_signal"].replace([np.inf, -np.inf], np.nan).dropna().max()
    if max_sig and max_sig > 0:
        grp["gap_signal_norm"] = (grp["gap_signal"] / max_sig * 100).round(2)
    else:
        grp["gap_signal_norm"] = 0.0

    return grp.sort_values("gap_signal_norm", ascending=False).reset_index(drop=True)
