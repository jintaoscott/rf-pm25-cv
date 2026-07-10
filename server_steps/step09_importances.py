"""
STEP 09 - Impurity-based feature importances at the two Optuna optima.

step07 produced SHAP importances only; Figure 6 of the manuscript is built
from IMPURITY importances (model.feature_importances_) of the Spatial-optimal
and Temporal-optimal Non-AOD models. This produces them. ~5 minutes.

Run AFTER step04 (uses config.OPTUNA_WINNERS, the step04 winners; update
src/config.py if step04 is ever rerun).

Outputs: results/importance_spatial.csv, results/importance_temporal.csv
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.data import load_dataset, prepare_features
from src import config

OPTUNA_PARAMS = config.OPTUNA_WINNERS  # single source: src/config.py

df = load_dataset(year=2019, include_aod=False)
dfx, features = prepare_features(df, "non_aod")
assert len(dfx) == 50551 and len(features) == 22, "wrong sample - ABORT"
X, y = dfx[features], dfx[config.TARGET]

for cv_name, params in OPTUNA_PARAMS.items():
    t0 = time.time()
    model = RandomForestRegressor(**params)
    model.fit(X, y)
    imp = pd.DataFrame({"feature": features,
                        "importance": model.feature_importances_}
                       ).sort_values("importance", ascending=False)
    out = config.RESULTS_DIR / f"importance_{cv_name}.csv"
    imp.to_csv(out, index=False)
    wshare = imp[imp.feature.isin(config.WEATHER_COLS)]["importance"].sum() * 100
    print(f"{cv_name}: saved {out.name}  weather share={wshare:.1f}%  "
          f"top3={list(imp.feature.head(3))}  ({time.time()-t0:.0f}s)", flush=True)

print("DONE")
