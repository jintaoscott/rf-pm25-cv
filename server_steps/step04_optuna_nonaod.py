"""
Joint hyperparameter optimization via Optuna: tests whether Spatial-CV and
Temporal-CV yield different optimal parameter sets under joint optimization.
Objective: the unweighted mean of the fold-specific R2 values across the ten
folds (the baselines and sweeps report pooled out-of-fold R2; see
src/block_cv.py). Fold assignment and RF fits use fixed seeds; the TPE
sampler is not seeded, so the archived trials in results/ are the record of
the search.

Non-AOD model, 2019 data, 100 trials each for Spatial-CV and Temporal-CV.
10-fold CV per trial.

Usage:
    pip install optuna
    python server_steps/step04_optuna_nonaod.py

Output: results/optuna_spatial.csv, results/optuna_temporal.csv, results/optuna_summary.txt
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import numpy as np
import pandas as pd

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from src.data import load_dataset, prepare_features
from src import config

N_SPLITS = 10
N_TRIALS = 100

print("Loading 2019 data...")
df = load_dataset(year=2019, include_aod=False)
dfx, features = prepare_features(df, "non_aod")
print(f"  {len(dfx)} rows, {len(features)} features")


def objective(trial, block_col):
    """Optuna objective: 10-fold block CV with sampled hyperparameters."""
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
        r2 = r2_score(dfx.loc[test, config.TARGET], y_pred)
        r2_scores.append(r2)

    return np.mean(r2_scores)


for cv_name, block_col in [("spatial", "index"), ("temporal", "random_dateid")]:
    print(f"\n{'='*60}")
    print(f"Optuna: {cv_name}-CV, {N_TRIALS} trials...")
    t0 = time.time()

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, block_col), n_trials=N_TRIALS)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")
    print(f"  Best R²: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Save all trials
    trials_df = study.trials_dataframe()
    trials_df.to_csv(config.RESULTS_DIR / f"optuna_{cv_name}.csv", index=False)

    # Save summary
    with open(config.RESULTS_DIR / "optuna_summary.txt", "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"{cv_name}-CV Optimization ({N_TRIALS} trials)\n")
        f.write(f"Best R²: {study.best_value:.4f}\n")
        f.write(f"Best params: {study.best_params}\n")
        f.write(f"Time: {elapsed:.0f}s\n")

print(f"\nResults saved to {config.RESULTS_DIR}")
