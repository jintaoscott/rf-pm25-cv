"""
Compute SHAP values for Non-AOD model with Optuna optimal params.
Two models: Spatial-CV-optimal and Temporal-CV-optimal.

Usage:
    python server_steps/step07_shap.py

Output: results/shap_spatial.csv, results/shap_temporal.csv
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

from src.data import load_dataset, prepare_features
from src import config

print("Loading 2019 data...", flush=True)
df = load_dataset(year=2019, include_aod=False)
dfx, features = prepare_features(df, "non_aod")
X = dfx[features]
y = dfx[config.TARGET]
print(f"  {len(X)} samples, {len(features)} features", flush=True)

# SHAP subsample: 500 points (TreeExplainer)
X_shap = X.sample(500, random_state=42)

OPTUNA_PARAMS = config.OPTUNA_WINNERS  # single source: src/config.py

for cv_name, params in OPTUNA_PARAMS.items():
    print(f"\n{'='*50}", flush=True)
    print(f"Training {cv_name}-optimal model...", flush=True)
    t0 = time.time()

    model = RandomForestRegressor(**params)
    model.fit(X, y)
    print(f"  R2: {model.score(X, y):.4f}", flush=True)

    print(f"  Computing SHAP (TreeExplainer, 500 samples)...", flush=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    importance = pd.DataFrame({
        'feature': features,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    outfile = config.RESULTS_DIR / f"shap_{cv_name}.csv"
    importance.to_csv(outfile, index=False)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s", flush=True)
    print(f"  Top 5: {list(importance.head(5)['feature'])}", flush=True)
    print(f"  Saved: {outfile}", flush=True)

print("\nAll done!", flush=True)
