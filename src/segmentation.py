"""
segmentation.py — K-Means segmentation pipeline for the Steam Market Intelligence Platform.

If a valid segmentation result already exists (persisted to data/segments.parquet),
it is loaded directly. Otherwise the pipeline runs and saves the result.

Cluster labels are assigned from actual cluster profiles — not hard-coded names.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import (
    SEGMENTATION_FEATURES,
    SEGMENTATION_K_RANGE,
    SEGMENTATION_K_DEFAULT,
    DATA_DIR,
)

log = logging.getLogger(__name__)
SEGMENTS_PATH = Path(DATA_DIR) / "segments.parquet"


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    """Select and clean segmentation features; return (clean_df, valid_index)."""
    available = [f for f in SEGMENTATION_FEATURES if f in df.columns]
    sub = df[available].fillna(df[available].median()).copy()
    return sub, sub.index


def run_elbow_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test k values in SEGMENTATION_K_RANGE.
    Returns DataFrame with columns: k, inertia, silhouette.
    """
    feat_df, _ = _prepare_features(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df)

    rows = []
    for k in SEGMENTATION_K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertia = km.inertia_
        sil = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=42)
        rows.append({"k": k, "inertia": inertia, "silhouette": sil})
        log.info(f"k={k}: inertia={inertia:.0f}, silhouette={sil:.4f}")
    return pd.DataFrame(rows)


def run_segmentation(df: pd.DataFrame, k: int | None = None, force: bool = False) -> pd.DataFrame:
    """
    Run K-Means segmentation and return the dataset with cluster labels.

    Parameters
    ----------
    df    : full dataset from data_loader
    k     : number of clusters (auto-selects best silhouette if None)
    force : re-run even if cached result exists

    Returns
    -------
    DataFrame with 'cluster_id' and 'cluster_label' columns appended.
    """
    if not force and SEGMENTS_PATH.exists():
        log.info(f"Loading cached segmentation from {SEGMENTS_PATH}")
        seg = pd.read_parquet(SEGMENTS_PATH)
        if "cluster_id" in seg.columns:
            return df.merge(seg[["app_id", "cluster_id", "cluster_label"]],
                            on="app_id", how="left")

    feat_df, valid_idx = _prepare_features(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df)

    # Choose k
    if k is None:
        elbow = run_elbow_analysis(df)
        best_row = elbow.loc[elbow["silhouette"].idxmax()]
        k = int(best_row["k"])
        log.info(f"Auto-selected k={k} (silhouette={best_row['silhouette']:.4f})")

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    result = df.loc[valid_idx].copy()
    result["cluster_id"] = labels

    # Build descriptive cluster profiles
    profile_cols = [c for c in SEGMENTATION_FEATURES if c in result.columns]
    profiles = result.groupby("cluster_id")[profile_cols].median()

    # Assign human-readable labels from actual data profiles
    result["cluster_label"] = result["cluster_id"].apply(
        lambda cid: _describe_cluster(cid, profiles)
    )

    # Persist
    SEGMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    result[["app_id", "cluster_id", "cluster_label"]].to_parquet(SEGMENTS_PATH, index=False)
    log.info(f"Segmentation saved to {SEGMENTS_PATH}")

    return result


def _describe_cluster(cid: int, profiles: pd.DataFrame) -> str:
    """
    Build a descriptive label from the cluster's median feature values.
    Labels are relative to other clusters — no invented marketing names.
    """
    row = profiles.loc[cid]

    # Rank clusters by price and quality (review_score_pct)
    price_rank   = profiles["price"].rank(ascending=False).astype(int).get(cid, "?") if "price" in profiles.columns else "?"
    quality_rank = profiles["review_score_pct"].rank(ascending=False).astype(int).get(cid, "?") if "review_score_pct" in profiles.columns else "?"

    price_level   = "High-Price"   if profiles.get("price", pd.Series()).rank(ascending=False).get(cid, 99) <= len(profiles) // 2 else "Low-Price"
    quality_level = "High-Quality" if profiles.get("review_score_pct", pd.Series()).rank(ascending=False).get(cid, 99) <= len(profiles) // 2 else "Lower-Quality"

    return f"Segment {cid + 1}: {price_level} / {quality_level}"


def get_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the median feature profile per cluster.
    Requires df to have 'cluster_id' column (from run_segmentation).
    """
    profile_cols = [c for c in SEGMENTATION_FEATURES if c in df.columns]
    return df.groupby(["cluster_id", "cluster_label"])[profile_cols].median().reset_index()
