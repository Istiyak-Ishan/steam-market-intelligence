"""
methodology.py — Analytical Methodology & Data Provenance documentation page.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd


def render(df: pd.DataFrame) -> None:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">📖 Analytical Methodology</div>
        <div class="hero-subtitle">
            Data provenance, feature engineering decisions, model training summary,
            and documented analytical caveats for the Steam Market Intelligence Platform.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_data, tab_features, tab_models, tab_signals, tab_caveats = st.tabs([
        "📦 Data & Provenance",
        "🔧 Feature Engineering",
        "🤖 ML Models",
        "📡 Analytical Signals",
        "⚠️ Caveats & Limits",
    ])

    # Dynamically compute dataset sizes
    cleaned_len = len(df)
    genre_counts = df["primary_genre"].value_counts()
    valid_genres = genre_counts[genre_counts >= 50].index
    eda_len = len(df[df["primary_genre"].isin(valid_genres)])
    
    # Accurate modeling subset filter (exclude Free, cap price, min reviews)
    modeling_len = len(df[(df["price"] > 0) & (df["price"] <= 80) & (df["total_review"] >= 5)])
    num_cols = len(df.columns)

    # ── DATA PROVENANCE ───────────────────────────────────────────────────────
    with tab_data:
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Raw Source Rows",    "136,971", help="Static pre-cleaned size from original dataset.")
        with c2: st.metric("Cleaned Dataset",    f"{cleaned_len:,}")
        with c3: st.metric("Modeling Subset",    f"{modeling_len:,}")
        with c4: st.metric("Feature Columns",    f"{num_cols:,}")

        st.markdown("""
        <div class="info-box">
            <strong>Source:</strong> Steam games dataset — all titles with a valid name, release date,
            and price field. Processing pipeline: null drop → feature extraction → type coercion →
            platform count computation → genre parsing → price tier assignment.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Row Count Reconciliation</div>', unsafe_allow_html=True)
        rec_data = {
            "Stage": [
                "Raw steam_games.csv",
                "After null name/date/price drop → steam_games_cleaned.csv",
                "Genre-filtered EDA subset (primary_genre ≥ 50 games)",
                "Modeling subset (paid, price $0.99–$80, ≥ 5 reviews)",
            ],
            "Row Count": ["136,971", f"{cleaned_len:,}", f"~{eda_len:,}", f"{modeling_len:,}"],
            "Used In": ["Source only", "All platform analyses", "EDA/reports only", "ML model training"],
        }
        st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Known Data Notes</div>', unsafe_allow_html=True)
        st.markdown("""
        | Note | Detail |
        |------|--------|
        | Column typo | `patforms_count` (not `platforms_count`) — matches training data, preserved intentionally |
        | `owners_mid` | `(lowest_estimate + highest_estimate) / 2` from Steam Spy bucket estimates |
        | `review_score_pct` | Ratio 0–1 (not percentage). Multiply by 100 for display. |
        | `value_score_calc` | `quality_score / price` — quality points per $1. Used by regression model. |
        | Free games | price = 0 → excluded from ML modeling subset |
        """, unsafe_allow_html=True)

    # ── FEATURE ENGINEERING ───────────────────────────────────────────────────
    with tab_features:
        st.markdown('<div class="section-header">Engineered Features</div>', unsafe_allow_html=True)

        feat_data = {
            "Feature": [
                "primary_genre", "price_tier", "quality_score", "value_score_calc",
                "log_reviews", "age_by_years", "is_indie", "genre_casual",
                "cat_single_player", "patforms_count", "release_year",
            ],
            "Source": [
                "genres (first listed)", "price bins", "review_score_pct × 100",
                "quality_score / price", "log1p(total_review)", "days_since_release / 365",
                "1 if 'Indie' in genres", "1 if 'Casual' in genres",
                "1 if 'Single-player' in categories", "len(platforms list)",
                "release_date year component",
            ],
            "Used In": [
                "All analyses", "All analyses", "ML input", "Regression target",
                "ML input", "ML input", "ML input", "ML input",
                "ML input", "Similarity / ML input", "Time series",
            ],
        }
        st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Similarity Features (Cosine Engine)</div>', unsafe_allow_html=True)
        st.markdown("""
        `price`, `review_score_pct`, `owners_mid`, `recommendations`, `peak_ccu`,
        `average_playtime_forever`, `languages_count`, `patforms_count`, `age_by_years`

        All features are **StandardScaler-normalised** before cosine similarity is computed.
        """)

    # ── ML MODELS ─────────────────────────────────────────────────────────────
    with tab_models:
        st.markdown('<div class="section-header">Price-Value Regressor</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.3rem;">Random Forest Regressor</div>
                <div class="metric-label">price_value_regressor.pkl</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.3rem;">Random Forest Classifier</div>
                <div class="metric-label">price_tier_classifier.pkl</div>
            </div>
            """, unsafe_allow_html=True)

        model_data = {
            "Property": ["Algorithm", "Training Set", "Target Variable",
                         "Input Features (count)", "Scaler at Inference", "Version Note"],
            "Regressor": ["RandomForestRegressor", "57,685 paid titles", "value_score_calc",
                          "12", "Not required (RF is scale-invariant)", "sklearn version mismatch expected"],
            "Classifier": ["RandomForestClassifier", "57,685 paid titles", "price_tier (4 classes)",
                           "12", "Not required (RF is scale-invariant)", "sklearn version mismatch expected"],
        }
        st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Model Input Features (exact order)</div>', unsafe_allow_html=True)
        st.markdown("""
        ```
        quality_score, age_by_years, categories_count, languages_count,
        peak_ccu, log_reviews, genre_casual, genre_count,
        full_audio_languages_count, is_indie, average_playtime_forever, cat_single_player
        ```
        """)

        st.markdown("""
        <div class="info-box">
            <strong>sklearn version mismatch:</strong> Models were trained on an older sklearn version.
            Predictions remain valid — Random Forest inference is not affected by minor version differences.
            One warning line per model load (suppressed thereafter).
        </div>
        """, unsafe_allow_html=True)

    # ── ANALYTICAL SIGNALS ────────────────────────────────────────────────────
    with tab_signals:
        st.markdown('<div class="section-header">Market Gap Signal</div>', unsafe_allow_html=True)
        st.markdown("""
        A transparent descriptive indicator — **not** a commercial opportunity guarantee.

        | Component | Formula |
        |-----------|---------|
        | Demand Indicator | `median_owners × (median_review_score / 100)` |
        | Supply Indicator | `game_count` in segment |
        | Gap Signal | `Demand / (1 + Supply)` → normalised 0–100 |

        Higher values = segments with higher observed demand relative to competing game count.
        """)

        st.markdown('<div class="section-header">Cosine Similarity Engine</div>', unsafe_allow_html=True)
        st.markdown("""
        1. StandardScaler normalises all 9 similarity features across the full dataset
        2. Cosine similarity computed between the query vector and all game vectors
        3. Results ranked by descending similarity score (0–1)
        4. The query game itself is excluded from results if it exists in the dataset
        """)

        st.markdown('<div class="section-header">K-Means Segmentation</div>', unsafe_allow_html=True)
        st.markdown("""
        - Algorithm: K-Means (scikit-learn), k=3–8 (default k=5)
        - Features: 9 standardised market metrics
        - Cluster labels: derived from actual median feature values — **no invented names**
        - Cache: `data/segments.parquet` (recomputed only when forced or k changes)
        """)

        st.markdown('<div class="section-header">Isolation Forest Anomaly Detection</div>', unsafe_allow_html=True)
        st.markdown("""
        - Algorithm: sklearn `IsolationForest` (n_estimators=100, random_state=42)
        - Features: price, review_score_pct, peak_ccu (log), average_playtime_forever (log), patforms_count, languages_count
        - Anomaly score: normalised to 0–100 (higher = more anomalous)
        - Contamination rate: user-configurable (default 1.5%)
        """)

    # ── CAVEATS ───────────────────────────────────────────────────────────────
    with tab_caveats:
        st.markdown('<div class="section-header">Documented Limitations</div>', unsafe_allow_html=True)
        st.markdown("""
        | Caveat | Detail |
        |--------|--------|
        | Correlation ≠ Causation | All correlations are observational. No causal claims are made. |
        | Ownership estimates | `owners_mid` is a bucket midpoint estimate (Steam Spy methodology), not an exact figure. |
        | Pricing data | Prices are at dataset snapshot time — not current live prices. |
        | ML predictions | Statistical associations from training data. Not a commercial guarantee. |
        | Genre classification | A game's `primary_genre` is the **first** genre listed in its Steam metadata. |
        | Free game exclusion | Free-to-play titles are excluded from ML model training (no meaningful value_score_calc). |
        | Market Gap Signal | Higher signal ≠ commercial opportunity. Supply/demand proxies only. |
        | Anomaly detection | Statistical outliers in feature space — does not imply quality judgements. |
        | Similarity engine | Cosine similarity in standardised feature space. Commercial proximity may differ. |
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            This platform is an analytical and research tool. All outputs are data-driven statistical
            descriptions of the Steam catalog at the time of data collection. No content constitutes
            financial advice, commercial recommendation, or guarantee of any outcome.
        </div>
        """, unsafe_allow_html=True)
