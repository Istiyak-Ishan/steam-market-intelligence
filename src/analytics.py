"""
analytics.py — Reusable analytical engine for the Steam Market Intelligence Platform.

All functions accept a DataFrame (from data_loader) and return clean DataFrames
or scalar values suitable for the Streamlit UI and visualization engine.

No fabricated numbers. All outputs are computed from the actual dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ════════════════════════════════════════════════════════════════════════════════
# ── GENRE ANALYTICS ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def genre_summary(df: pd.DataFrame, min_games: int = 50) -> pd.DataFrame:
    """
    Aggregate key market metrics by primary_genre.

    Returns a DataFrame indexed by primary_genre with columns:
      game_count, mean_price, median_price, mean_quality, median_quality,
      mean_owners, median_owners, mean_reviews, median_reviews,
      mean_peak_ccu, mean_playtime_hours, mean_languages, mean_platforms,
      mean_recommendations
    """
    df = df[df["primary_genre"].notna()].copy()
    grp = df.groupby("primary_genre")

    agg = grp.agg(
        game_count=("app_id", "count"),
        mean_price=("price", "mean"),
        median_price=("price", "median"),
        mean_quality=("review_score_pct", lambda x: x.dropna().mean() * 100),
        median_quality=("review_score_pct", lambda x: x.dropna().median() * 100),
        mean_owners=("owners_mid", "mean"),
        median_owners=("owners_mid", "median"),
        mean_reviews=("total_review", "mean"),
        median_reviews=("total_review", "median"),
        mean_peak_ccu=("peak_ccu", "mean"),
        mean_playtime_hrs=("average_playtime_forever", lambda x: x.mean() / 60),
        mean_languages=("languages_count", "mean"),
        mean_platforms=("patforms_count", "mean"),
        mean_recommendations=("recommendations", "mean"),
    ).reset_index()

    agg = agg[agg["game_count"] >= min_games].copy()
    return agg.sort_values("game_count", ascending=False).reset_index(drop=True)


def genre_price_summary(df: pd.DataFrame, min_games: int = 50) -> pd.DataFrame:
    """Mean and median price per genre. Returns sorted DataFrame."""
    df = df[df["primary_genre"].notna()].copy()
    grp = df.groupby("primary_genre").agg(
        game_count=("app_id", "count"),
        mean_price=("price", "mean"),
        median_price=("price", "median"),
        std_price=("price", "std"),
        p25_price=("price", lambda x: x.quantile(0.25)),
        p75_price=("price", lambda x: x.quantile(0.75)),
    ).reset_index()
    grp = grp[grp["game_count"] >= min_games]
    return grp.sort_values("mean_price", ascending=False).reset_index(drop=True)


def genre_quality_summary(df: pd.DataFrame, min_games: int = 50) -> pd.DataFrame:
    """Mean and median review quality per genre."""
    df = df[df["primary_genre"].notna() & df["review_score_pct"].notna()].copy()
    grp = df.groupby("primary_genre").agg(
        game_count=("app_id", "count"),
        mean_quality=("review_score_pct", lambda x: x.mean() * 100),
        median_quality=("review_score_pct", lambda x: x.median() * 100),
    ).reset_index()
    grp = grp[grp["game_count"] >= min_games]
    return grp.sort_values("mean_quality", ascending=False).reset_index(drop=True)


def genre_ownership_summary(df: pd.DataFrame, min_games: int = 50) -> pd.DataFrame:
    """Ownership metrics by genre."""
    df = df[df["primary_genre"].notna()].copy()
    grp = df.groupby("primary_genre").agg(
        game_count=("app_id", "count"),
        mean_owners=("owners_mid", "mean"),
        median_owners=("owners_mid", "median"),
        mean_recommendations=("recommendations", "mean"),
        mean_peak_ccu=("peak_ccu", "mean"),
    ).reset_index()
    grp = grp[grp["game_count"] >= min_games]
    return grp.sort_values("mean_owners", ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════════
# ── PRICE ANALYTICS ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def price_distribution(df: pd.DataFrame, max_price: float = 80.0) -> pd.Series:
    """Return price Series filtered to [0, max_price] for distribution plots."""
    return df.loc[df["price"] <= max_price, "price"]


def price_percentiles(df: pd.DataFrame) -> pd.Series:
    """Return price percentiles at 10% intervals for the full dataset."""
    prices = df["price"].dropna()
    percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    return pd.Series(
        {f"p{p}": prices.quantile(p / 100) for p in percentiles}
    )


def price_tier_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage by price_tier."""
    tier_order = ["Free", "Budget", "Mid-range", "Premium", "AAA"]
    counts = df["price_tier"].value_counts().reindex(tier_order, fill_value=0)
    pct = (counts / counts.sum() * 100).round(2)
    return pd.DataFrame({"count": counts, "pct": pct}).reset_index().rename(
        columns={"index": "price_tier"}
    )


def price_vs_quality(df: pd.DataFrame, min_reviews: int = 5) -> pd.DataFrame:
    """
    Return a scatter-ready DataFrame: price, review_score_pct, owners_mid,
    primary_genre, name. Filtered to paid games with sufficient reviews.
    """
    mask = (
        (df["price"] > 0) &
        (df["total_review"] >= min_reviews) &
        df["review_score_pct"].notna()
    )
    cols = ["name", "price", "review_score_pct", "owners_mid",
            "primary_genre", "price_tier", "total_review"]
    return df.loc[mask, [c for c in cols if c in df.columns]].copy()


def price_vs_ownership(df: pd.DataFrame) -> pd.DataFrame:
    """Return price × owners_mid scatter data for paid games."""
    mask = (df["price"] > 0) & df["owners_mid"].notna()
    return df.loc[mask, ["name", "price", "owners_mid", "primary_genre",
                          "price_tier", "review_score_pct"]].copy()


# ════════════════════════════════════════════════════════════════════════════════
# ── QUALITY ANALYTICS ──────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def quality_distribution(df: pd.DataFrame) -> pd.Series:
    """Review score percentage distribution for games with ≥1 review."""
    return df.loc[df["has_reviews"] == True, "review_score_pct"].dropna()


def quality_by_price_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean and median review score per price tier.
    Excludes games with no reviews.
    """
    tier_order = ["Free", "Budget", "Mid-range", "Premium", "AAA"]
    df_r = df[df["has_reviews"] == True].copy()
    grp = df_r.groupby("price_tier").agg(
        game_count=("app_id", "count"),
        mean_quality=("review_score_pct", lambda x: x.mean() * 100),
        median_quality=("review_score_pct", lambda x: x.median() * 100),
    ).reindex(tier_order)
    return grp.reset_index()


def quality_percentiles(df: pd.DataFrame) -> pd.Series:
    """Percentiles of review_score_pct for games with ≥1 review."""
    scores = df.loc[df["has_reviews"] == True, "review_score_pct"].dropna()
    percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    return pd.Series(
        {f"p{p}": scores.quantile(p / 100) * 100 for p in percentiles}
    )


# ════════════════════════════════════════════════════════════════════════════════
# ── POPULARITY ANALYTICS ───────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def ownership_distribution(df: pd.DataFrame) -> pd.Series:
    """owners_mid distribution (log-scale recommended for plotting)."""
    return df["owners_mid"].dropna()


def top_games_by_owners(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return top N games by estimated ownership midpoint, breaking ties by review engagement."""
    cols = ["name", "owners_mid", "price", "review_score_pct",
            "primary_genre", "price_tier", "app_id", "total_review", "positive"]
    available = [c for c in cols if c in df.columns]
    sort_cols = [c for c in ["owners_mid", "total_review", "positive"] if c in df.columns]
    return (
        df[available]
        .dropna(subset=["owners_mid"])
        .sort_values(sort_cols, ascending=[False] * len(sort_cols))
        .head(n)
        .reset_index(drop=True)
    )


def top_games_by_reviews(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return top N games by total review count."""
    cols = ["name", "total_review", "review_score_pct", "price",
            "primary_genre", "price_tier", "app_id"]
    available = [c for c in cols if c in df.columns]
    return (
        df[available]
        .sort_values("total_review", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def top_games_by_ccu(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return top N games by peak concurrent users."""
    cols = ["name", "peak_ccu", "price", "review_score_pct",
            "primary_genre", "price_tier", "app_id"]
    available = [c for c in cols if c in df.columns]
    return (
        df[available]
        .sort_values("peak_ccu", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


# ════════════════════════════════════════════════════════════════════════════════
# ── ENGAGEMENT ANALYTICS ───────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def engagement_by_genre(df: pd.DataFrame, min_games: int = 50) -> pd.DataFrame:
    """
    Playtime and engagement metrics by genre.
    Uses average_playtime_forever (in minutes; convert to hours for display).
    """
    df = df[df["primary_genre"].notna()].copy()
    grp = df.groupby("primary_genre").agg(
        game_count=("app_id", "count"),
        mean_avg_playtime_hrs=("average_playtime_forever", lambda x: x.mean() / 60),
        median_avg_playtime_hrs=("average_playtime_forever", lambda x: x.median() / 60),
        mean_peak_ccu=("peak_ccu", "mean"),
    ).reset_index()
    grp = grp[grp["game_count"] >= min_games]
    return grp.sort_values("mean_avg_playtime_hrs", ascending=False).reset_index(drop=True)


def playtime_distribution(df: pd.DataFrame, max_hours: float = 200.0) -> pd.Series:
    """average_playtime_forever in hours, capped at max_hours."""
    hrs = df["average_playtime_forever"].dropna() / 60
    return hrs[hrs <= max_hours]


# ════════════════════════════════════════════════════════════════════════════════
# ── PLATFORM ANALYTICS ─────────────────────────────────────────────════════════
# ════════════════════════════════════════════════════════════════════════════════

def platform_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Percentage of games supporting Windows / Mac / Linux.
    Returns a 3-row DataFrame: platform, count, pct.
    """
    total = len(df)
    rows = []
    for plat in ["windows", "mac", "linux"]:
        if plat in df.columns:
            n = int(df[plat].fillna(0).sum())
            rows.append({"platform": plat.title(), "count": n, "pct": n / total * 100})
    return pd.DataFrame(rows)


def platform_count_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of games by number of supported platforms."""
    counts = df["patforms_count"].fillna(0).astype(int).value_counts().sort_index()
    total = counts.sum()
    return pd.DataFrame({
        "platform_count": counts.index,
        "games": counts.values,
        "pct": (counts.values / total * 100).round(2),
    })


def platform_vs_ownership(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ownership by platform count category."""
    df = df.copy()
    df["plat_grp"] = df["patforms_count"].fillna(0).astype(int).astype(str) + " platform(s)"
    return (
        df.groupby("plat_grp")
        .agg(
            game_count=("app_id", "count"),
            mean_owners=("owners_mid", "mean"),
            median_owners=("owners_mid", "median"),
            mean_quality=("review_score_pct", lambda x: x.dropna().mean() * 100),
        )
        .reset_index()
        .sort_values("plat_grp")
    )


# ════════════════════════════════════════════════════════════════════════════════
# ── LOCALIZATION ANALYTICS ─────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def language_count_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Grouped distribution of language support breadth."""
    bins   = [-1, 1, 5, 10, 25, 200]
    labels = ["1 lang", "2–5 langs", "6–10 langs", "11–25 langs", ">25 langs"]
    df = df.copy()
    df["lang_group"] = pd.cut(df["languages_count"], bins=bins, labels=labels)
    counts = df["lang_group"].value_counts(sort=False)
    return pd.DataFrame({"group": counts.index.astype(str), "count": counts.values})


def language_vs_ownership(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ownership by language support group."""
    bins   = [-1, 1, 5, 10, 25, 200]
    labels = ["1 lang", "2–5 langs", "6–10 langs", "11–25 langs", ">25 langs"]
    df = df.copy()
    df["lang_group"] = pd.cut(df["languages_count"], bins=bins, labels=labels)
    grp = df.groupby("lang_group", observed=False).agg(
        game_count=("app_id", "count"),
        mean_owners=("owners_mid", "mean"),
        median_owners=("owners_mid", "median"),
        pearson_corr=("owners_mid", lambda x: x.corr(df.loc[x.index, "languages_count"])),
    ).reset_index()
    grp["lang_group"] = grp["lang_group"].astype(str)
    return grp


def language_count_by_genre(df: pd.DataFrame, min_games: int = 50) -> pd.DataFrame:
    """Mean language count per primary genre."""
    df = df[df["primary_genre"].notna()].copy()
    grp = df.groupby("primary_genre").agg(
        game_count=("app_id", "count"),
        mean_languages=("languages_count", "mean"),
        median_languages=("languages_count", "median"),
    ).reset_index()
    return grp[grp["game_count"] >= min_games].sort_values(
        "mean_languages", ascending=False
    ).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════════
# ── TIME ANALYTICS ─────────────────────────────────────────════════════════════
# ════════════════════════════════════════════════════════════════════════════════

def release_volume_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Game release count per year."""
    if "release_year" not in df.columns:
        from src.feature_engineering import add_release_year
        df = add_release_year(df)
    counts = (
        df.dropna(subset=["release_year"])
        .groupby("release_year")
        .size()
        .reset_index(name="game_count")
    )
    counts["release_year"] = counts["release_year"].astype(int)
    return counts.sort_values("release_year")


def price_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """Median and mean price per release year (paid games only)."""
    if "release_year" not in df.columns:
        from src.feature_engineering import add_release_year
        df = add_release_year(df)
    paid = df[df["price"] > 0].dropna(subset=["release_year"]).copy()
    grp = paid.groupby("release_year").agg(
        game_count=("app_id", "count"),
        mean_price=("price", "mean"),
        median_price=("price", "median"),
    ).reset_index()
    grp["release_year"] = grp["release_year"].astype(int)
    return grp.sort_values("release_year")


def quality_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """Median review score per release year."""
    if "release_year" not in df.columns:
        from src.feature_engineering import add_release_year
        df = add_release_year(df)
    reviewed = df[df["has_reviews"] == True].dropna(subset=["release_year", "review_score_pct"]).copy()
    grp = reviewed.groupby("release_year").agg(
        game_count=("app_id", "count"),
        mean_quality=("review_score_pct", lambda x: x.mean() * 100),
        median_quality=("review_score_pct", lambda x: x.median() * 100),
    ).reset_index()
    grp["release_year"] = grp["release_year"].astype(int)
    return grp.sort_values("release_year")


# ════════════════════════════════════════════════════════════════════════════════
# ── CORRELATION ANALYTICS ──────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

CORR_NUMERIC_COLS = [
    "price", "review_score_pct", "owners_mid",
    "recommendations", "peak_ccu",
    "average_playtime_forever", "languages_count", "patforms_count",
    "total_review", "age_by_years",
]


def correlation_matrix(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute pairwise correlation matrix.

    Parameters
    ----------
    cols   : list of column names; defaults to CORR_NUMERIC_COLS
    method : 'pearson' or 'spearman'
    """
    if cols is None:
        cols = [c for c in CORR_NUMERIC_COLS if c in df.columns]
    subset = df[cols].dropna()
    return subset.corr(method=method)


def key_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business-relevant pairwise correlations (Pearson + Spearman).
    Returns a DataFrame with columns: feature_a, feature_b, pearson, spearman.
    """
    pairs = [
        ("price",             "review_score_pct"),
        ("price",             "owners_mid"),
        ("price",             "total_review"),
        ("review_score_pct",  "owners_mid"),
        ("review_score_pct",  "recommendations"),
        ("peak_ccu",          "owners_mid"),
        ("languages_count",   "owners_mid"),
        ("patforms_count",    "owners_mid"),
        ("age_by_years",      "price"),
        ("average_playtime_forever", "review_score_pct"),
    ]
    rows = []
    for a, b in pairs:
        if a not in df.columns or b not in df.columns:
            continue
        sub = df[[a, b]].dropna()
        if len(sub) < 10:
            continue
        r_p = sub[a].corr(sub[b], method="pearson")
        r_s = sub[a].corr(sub[b], method="spearman")
        rows.append({
            "feature_a": a, "feature_b": b,
            "pearson": round(r_p, 4), "spearman": round(r_s, 4),
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════════
# ── SUMMARY STATS ──────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def dataset_summary(df: pd.DataFrame) -> dict:
    """High-level summary statistics for the Overview page."""
    paid = df[df["price"] > 0]
    with_reviews = df[df["has_reviews"] == True]
    return {
        "total_games":            len(df),
        "unique_genres":          df["primary_genre"].nunique(),
        "paid_games":             len(paid),
        "free_games":             int((df["price"] == 0).sum()),
        "median_price":           float(paid["price"].median()),
        "mean_price":             float(paid["price"].mean()),
        "median_review_pct":      float(with_reviews["review_score_pct"].median() * 100) if len(with_reviews) > 0 else None,
        "total_estimated_owners": float(df["owners_mid"].sum()),
        "year_range":             (
            int(df["release_year"].dropna().min()),
            int(df["release_year"].dropna().max()),
        ) if "release_year" in df.columns else None,
        "pct_windows":            float(df["windows"].fillna(0).mean() * 100),
        "pct_mac":                float(df["mac"].fillna(0).mean() * 100),
        "pct_linux":              float(df["linux"].fillna(0).mean() * 100),
        "pct_indie":              float(df["genres"].fillna("").str.contains("Indie", case=False).mean() * 100),
        "median_languages":       float(df["languages_count"].median()),
        "games_with_reviews":     len(with_reviews),
    }
