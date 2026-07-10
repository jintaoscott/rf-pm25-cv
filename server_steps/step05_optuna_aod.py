"""
Joint hyperparameter optimization (Optuna) for the AOD model, sentinel-full sample.

Joint optimization for the AOD model (24 features, 99999 sentinels kept),
complementing the Non-AOD optimization in step04.
Objective: the unweighted mean of the fold-specific R2 values across the ten
folds, as in step04 (fold and RF seeds fixed; TPE sampler not seeded).

AOD model, 2019 data, 100 trials each for Spatial-CV and Temporal-CV, 10-fold per trial.

Usage:
    python server_steps/step05_optuna_aod.py

Output: results/optuna_aod_spatial.csv, results/optuna_aod_temporal.csv,
        appended summary in results/optuna_aod_summary.txt
Expected runtime: roughly comparable to the Non-AOD run (~10-25 min per CV).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from src.data import load_dataset, prepare_features
from src import config

SENTINEL = 99999
N_SPLITS = 10
N_TRIALS = 100

print("Loading 2019 data (AOD model, sentinel-full)...")
df = load_dataset(year=2019, include_aod=True)
dfx, features = prepare_features(df, "aod")
n = len(dfx)
pct_missing = 100.0 * (dfx["AOD"].values == SENTINEL).mean()
print(f"  {n} rows, {len(features)} features | Terra sentinel share {pct_missing:.1f}%")
assert n > 50000 and pct_missing > 50, "not the sentinel-full sample - ABORT"


def objective(trial, block_col):
    params = {
        "n_estimators": 90,
        "max_features": trial.suggest_int("max_features", 1, len(features)),
        "max_samples": trial.suggest_float("max_samples", 0.02, 1.0),
        "max_depth": trial.suggest_int("max_depth", 5, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 35),
        "random_state": 42,
        "n_jobs": -1,
    }
    blocks = dfx[block_col].unique()
    np.random.seed(42)
    np.random.shuffle(blocks)
    folds = np.array_split(blocks, N_SPLITS)
    r2_scores = []
    for test_blocks in folds:
        train = ~dfx[block_col].isin(test_blocks)
        test = dfx[block_col].isin(test_blocks)
        model = RandomForestRegressor(**params)
        model.fit(dfx.loc[train, features], dfx.loc[train, config.TARGET])
        y_pred = model.predict(dfx.loc[test, features])
        r2_scores.append(r2_score(dfx.loc[test, config.TARGET], y_pred))
    return np.mean(r2_scores)


for cv_name, block_col in [("spatial", "index"), ("temporal", "random_dateid")]:
    print(f"\n{'='*60}")
    print(f"Optuna (AOD model): {cv_name}-CV, {N_TRIALS} trials...")
    t0 = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, block_col), n_trials=N_TRIALS)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s | Best R²: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    study.trials_dataframe().to_csv(config.RESULTS_DIR / f"optuna_aod_{cv_name}.csv", index=False)
    with open(config.RESULTS_DIR / "optuna_aod_summary.txt", "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"AOD model (sentinel-full) {cv_name}-CV ({N_TRIALS} trials)\n")
        f.write(f"Best R²: {study.best_value:.4f}\nBest params: {study.best_params}\n")
        f.write(f"Time: {elapsed:.0f}s\n")

print(f"\nResults saved to {config.RESULTS_DIR}")
