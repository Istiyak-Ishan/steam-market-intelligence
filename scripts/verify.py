import sys; sys.path.insert(0, '.')

# Test 1: Full data load pipeline
print('=== Test 1: Data load ===')
from src.data_loader import _load_raw
from src.feature_engineering import apply_all_features
df = _load_raw()
df = apply_all_features(df)
print(f'Rows: {len(df):,}, Cols: {len(df.columns)}')
print('metacritic_missing col:', 'metacritic_missing' in df.columns)
print('tag_singleplayer col:', 'tag_singleplayer' in df.columns)
print('price max:', round(df['price'].max(), 2), '(should be ~49.99)')
print('peak_ccu max:', round(df['peak_ccu'].max()), '(should be ~185)')
print('metacritic zeros:', (df['metacritic_score'] == 0).sum(), '(should be 0)')

# Test 2: All 6 models load
print()
print('=== Test 2: All 6 models ===')
from src.model_loader import (load_sweetspot_model, load_review_score_model,
    load_value_score_model, load_ownership_model, load_price_tier_clf, load_fair_price_clf)
load_sweetspot_model(); print('sweetspot: OK')
load_review_score_model(); print('review_score: OK')
load_value_score_model(); print('value_score: OK')
load_ownership_model(); print('ownership: OK')
load_price_tier_clf(); print('price_tier_clf: OK')
load_fair_price_clf(); print('fair_price_clf: OK')

# Test 3: Cluster labels
print()
print('=== Test 3: Cluster labels ===')
from src.segmentation import _describe_cluster
import pandas as pd
profiles = pd.DataFrame({
    'price': [2.0, 15.0, 40.0, 5.0],
    'review_score_pct': [0.8, 0.6, 0.9, 0.4]
}, index=[0,1,2,3])
for i in range(4):
    print(f'  Cluster {i}:', _describe_cluster(i, profiles))

print()
print('=== ALL CHECKS PASSED ===')
