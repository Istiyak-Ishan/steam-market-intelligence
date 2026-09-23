# Steam Market Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://steam-market-intelligence-qikxtydhwaitfrgdgfncfk.streamlit.app/)

An empirically driven, machine-learning-backed analytical platform designed to help Indie developers and AA publishers enter the Steam marketplace with data-backed strategies.

The platform provides a complete **Cyberpunk UI** aesthetic, engineered for high-performance interactivity without browser lockups, seamlessly analyzing over 74,000 commercially released titles from 1997–2024.

---

## Key Features

### 1. Market Insights
- **Overview & Market Explorer:** Macro-level EDA of Steam's history. High-performance scatter/bubble charts capped dynamically to prevent browser freezing.
- **Genre Benchmark:** Deep dive into specific genres to find median price points, value scores, and engagement rates.
- **Anomaly Finder:** Discover statistical outliers — games that defied the odds with massive CCUs despite high prices or low review scores.

### 2. Machine Learning Tools
- **Predict Tool:** A powerful 3-in-1 prediction engine. Input hypothetical game metrics to instantly receive:
  - **Predicted Value Score** (Quality Points per $1)
  - **Predicted Price Tier** (Budget, Mid-range, Premium, AAA)
  - **Fair Price Verdict** (Is the game overpriced for its genre?)
- **Model Inspector:** Transparent ML diagnostics computed *live* on the dataset. View Feature Importance (Tree-based & Permutation) and live confusion matrices without relying on static files.
- **Cluster Explorer:** Explore hidden market archetypes using K-Means unsupervised learning.

### 3. Game Intelligence
- **Game Analyzer & Comparison:** Head-to-head radar charts and deep metric profiling. Searches use optimized top-15k native dropdowns to maintain snappy UX.
- **Similar Games Engine:** Find direct competitors using a 9-feature standardized Cosine Similarity algorithm.
- **Publisher Studio:** Input a 'Base Scenario' and run 'What-If' analyses to instantly predict shifts in Price Tier and Value Score, visualized via Waterfall charts.
- **Strategic Playbook:** Generates an automated, textual, and actionable business plan based on your exact game parameters.

### 4. Technical Excellence
- **Cyberpunk Design System:** A highly polished, custom CSS system utilizing `JetBrains Mono` for data and `IBM Plex Sans` for body text, styled with deep blacks (`#05050a`), neon cyan (`#00f5ff`), and vivid pink (`#ff0066`).
- **Memory Optimized:** Data pipelines utilize `@st.cache_resource` for zero-copy memory reads. UI elements employ intelligent data culling to keep the DOM light and snappy.

---

## Tech Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/) + Vanilla CSS injected via `st.markdown`
- **Data Visualizations:** [Plotly Express & Graph Objects](https://plotly.com/python/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (`HistGradientBoostingRegressor/Classifier`, `DecisionTree`, `KMeans`)
- **Data Processing:** [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)

---

## Repository Structure

```text
steam-market-intelligence/
├── app.py                     # Streamlit entry point & sidebar routing
├── app/
│   └── pages/                 # Individual Streamlit pages (Predict Tool, Playbook, etc.)
├── src/                       # Core backend logic
│   ├── config.py              # Centralized constants and UI tokens
│   ├── data_loader.py         # Memory-cached data ingestion
│   ├── feature_engineering.py # Data transforms and pipelines
│   ├── model_loader.py        # ML model serving
│   └── similarity.py          # Nearest-neighbor algorithms
├── models/                    # Trained .pkl models (Sweetspot, Review Score, Tiers, etc.)
├── data/                      # Cleaned dataset (steam_games_cleaned.csv)
├── scripts/
│   └── train_pipeline.py      # Offline ML training script
└── notebooks/                 # Original EDA Jupyter notebooks
```

---

## Installation & Usage

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/Istiyak-Ishan/steam-market-intelligence.git
cd steam-market-intelligence
```

### Windows
```powershell
# Create a virtual environment
python -m venv venv

# Activate the environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py
```

### Linux / macOS
```bash
# Create a virtual environment (you may need to use python3)
python3 -m venv venv

# Activate the environment
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Run the app locally
streamlit run app.py
```

---

## Retraining the Models
If you update `steam_games_cleaned.csv` with newer data, you must retrain the ML pipeline:
```bash
python scripts/train_pipeline.py
```
This will automatically re-engineer features, train the 6 distinct models, and output fresh `.pkl` files to the `models/` directory.

---

## Author

**Istiyak Hossain Ishan**
