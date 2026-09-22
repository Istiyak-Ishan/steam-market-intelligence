# Steam Market Intelligence

An empirically driven, machine-learning-backed analytical tool designed to help Indie developers and AA publishers enter the Steam marketplace with data-backed strategies.

## Features

- **Macro Exploratory Data Analysis (EDA):** Gain high-level insights into Steam's entire history, spanning over 136,000 titles from 1997-2024.
- **Machine Learning Inference:** Random Forest models predict a title's natural price tier and expected value score based on game features.
- **Publisher Scenario Engine:** Input a 'Base Scenario' and run 'What-If' analyses to instantly predict shifts in Price Tier and Value Score.
- **Similarity Engine:** Discover direct competitors using a 9-feature standardized Cosine Similarity Engine.
- **Market Segmentation:** Explore hidden market archetypes clustered by K-Means unsupervised learning.
- **Gaming HUD Aesthetic:** A polished, tactical, dark-mode user interface designed for immediate clarity and impact.

## Installation

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
```

### Linux / macOS
```bash
# Create a virtual environment (you may need to use python3)
python3 -m venv venv

# Activate the environment
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
```

## Usage

Run the Streamlit application locally:

```bash
streamlit run app.py
```

## Tech Stack

- **Python** for core logic
- **Streamlit** for the frontend application and interactive UI
- **Plotly** for high-performance, interactive data visualization
- **Scikit-Learn** for Machine Learning models (Random Forest, K-Means clustering, Cosine Similarity)
- **Pandas & NumPy** for data manipulation

## Author

**Istiyak Hossain Ishan**
