# TASK_CHECKLIST.md
# Steam Market Intelligence Platform — Task Checklist

## Day 1 ✅ COMPLETE
- [x] Inspect full repo, notebooks, CSVs, PKL artifacts
- [x] Build internal map: data → preprocessing → features → models
- [x] Implement all 11 src/ modules
- [x] Implement all 7 Streamlit app pages
- [x] Fix similarity.py column merge bug
- [x] Fix segmentation.py IntCastingNaNError
- [x] Fix model_loader.py warning spam (deduplicate per load)
- [x] Fix market_explorer.py inline imports
- [x] Write tests/test_day1.py — 29 tests, all passing
- [x] Write PROJECT_STATUS.md and TASK_CHECKLIST.md

---

## Day 2 — All 7 Tasks COMPLETE ✅

### Approved Implementation Order:
1. [x] **segmentation_page.py** — Cluster explorer, 3D PCA scatter, profile breakdown, cluster summary table. (DONE)
2. [x] **model_lab.py** — Live RF Value Score regressor & Price Tier classifier interactive lab, feature importance, confusion matrix, residual analysis. (DONE)
3. [x] **publisher_studio.py** — Indie Developer Studio & What-If Scenario Simulator: interactive sliders/presets, live ML predictions, genre median comparison radar, fair-price decision support with price elasticity simulator, similar published games reference. (DONE)
4. [x] **genre_benchmark.py** — Deep Genre Benchmarks: competitive density heatmap (Genre × Price Tier), market gap signals (demand vs supply), pricing/engagement distributions, annual release density trends, top benchmark export. (DONE)
5. [x] **game_comparison.py** — Multi-game head-to-head comparison engine: side-by-side KPI cards, overlay radar chart, direct metric bar comparisons, price-quality matrix quadrant view, ML value score/tier predictions, and spec sheet export. (DONE)
6. [x] **similar_games.py** — Dedicated Similar Game Finder & Competitor Intelligence: dual-mode lookup (existing titles or custom query profile), filterable cosine similarity, target-vs-peer overlay radar, competitor feature space scatter, pricing/reception gap analysis, CSV export. (DONE)
7. [x] **anomaly_finder.py** — Statistical & ML Anomaly Finder: identify pricing anomalies, hidden gems, viral outliers, engagement sleepers, and unsupervised Isolation Forest multidimensional outliers. (DONE)

---

## Day 3 — Design System, Navigation & App Shell ✅ COMPLETE

- [x] Full CSS design system in app.py
- [x] Sidebar redesigned with grouped navigation
- [x] Plotly theme tokens updated in src/config.py
- [x] methodology.py created (5-tab provenance page)
- [x] 51/51 tests pass

---

## Day 3 Content — Overview & Market Explorer ✅ COMPLETE

### Overview page (`app/pages/overview.py`)
- [x] Hero section with gradient title, platform label, dynamic badge
- [x] 6 live KPIs: total games, year span, genres, median price, median review, median ownership
- [x] 6 Market Signals (all computed from live df, no hardcoded values):
  - Indie vs non-indie quality gap
  - Price↔ownership Spearman correlation
  - Free-to-play ownership premium multiplier
  - Localization ownership multiplier (10+ langs vs ≤2)
  - Review-quality ownership uplift (≥80% vs <50%)
  - Indie market share %
- [x] 5 analytical charts: annual releases area, price tier donut, top genres bar, pricing timeline, quality timeline
- [x] Genre landscape scatter: mean price vs quality, bubble size=game count, colour=ownership
- [x] Dataset provenance note (live count, not hardcoded)

### Market Explorer (`app/pages/market_explorer.py`)
- [x] 8 sidebar filters: genre (multi), price tier (multi), price range, year range, review min, platform, language min, ownership min
- [x] All filters apply simultaneously to all charts
- [x] Price slider upper bound auto-extends to actual dataset max (avoids capping at $100)
- [x] Reset Filters button
- [x] Filter summary bar showing N/total retained with %
- [x] 8 charts across 4 tabs:
  - Tab 1: Genre×Price boxplot + Price histogram + Genre×Tier heatmap
  - Tab 2: Price vs Quality scatter (coloured by tier) + Genre Value Ranking bar
  - Tab 3: Ownership Bubble chart + Genre ownership table + Top 15 owners
  - Tab 4: Genre Release Timeline (stacked area) + Cohort Heatmap (year×tier)
- [x] All charts verified with plotly.graph_objects.Figure type assertions
- [x] 51/51 tests still passing

## Architecture Notes (for next session)

### How to start the app
`
streamlit run app.py
`

### How to run tests
`
python -m pytest tests/test_day1.py -v
`

### Key API contracts (DO NOT break these)
- src/model_loader.py → predict_value_score(profile: dict) -> float
- src/model_loader.py → predict_price_tier(profile: dict) -> tuple[str, dict]
- src/similarity.py → ind_similar_games(game_profile, df, n) -> DataFrame
- src/benchmarks.py → enchmark_game(row, df) -> dict with keys: game_name, genre, price_tier, market_percentiles, genre_medians
- src/validation.py → un_all_validations(df) -> dict with keys: ok, errors, warnings
- Column typo is INTENTIONAL: patforms_count (matches training data)

### Model feature order (MODEL_FEATURES from src/config.py)
price, review_score_pct, owners_mid, recommendations, peak_ccu,
average_playtime_forever, languages_count, patforms_count, age_by_years
(RF models do NOT need StandardScaler at inference)

---

## Day 3 Content — Final QA & Strategic Playbook ✅ COMPLETE

- [x] Create `app/pages/strategic_playbook.py` with 4 tailored audience sections (Gamers, Indie Developers, Publishers, Platform Analysts)
- [x] Integrate Strategic Playbook into `app.py` routing (under Overview group)
- [x] Run full QA sweep and verify all `tests/` pass (51/51)
- [x] Fix broken tests in `test_day2.py` resulting from Publisher Studio rewrite
- [x] Finalize `PROJECT_STATUS.md`

---

## Phase 4: Exhaustive UI/UX & Functional Stress Test ✅ COMPLETE

- [x] Deep audit of `app.py` and Streamlit UI design system.
- [x] Validate Market Explorer edge cases ($0 F2P pricing, extreme outliers).
- [x] Refine Plotly visual constraints (fix text collisions, overlap).
- [x] Fix discretized y-axis banding with continuous density jitter.
- [x] Apply exponential contrast metrics to bubble charts.
- [x] Prevent categorical color overload in large scatter plots.
