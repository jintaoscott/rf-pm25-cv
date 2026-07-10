"""
Experiment 2: Hyperparameter sweep (max_features, max_samples) for all years.
Non-AOD model only, Spatial + Temporal CV, 10-fold × 5 iterations.

Tests whether the hyperparameter trends hold in each year 2015-2019.

Usage:
    python server_steps/step08_multiyear_sweeps.py

Output: results/sweep_{year}_{hp}_{cv}.csv for each combination
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from src.data import load_dataset, prepare_features
from src.block_cv import BlockCV
from src import config

import pandas as pd
import numpy as np

N_SPLITS = 10
N_ITER = 5  # iterations per parameter value

YEARS = [2015, 2016, 2017, 2018, 2019]
HYPERPARAMS = ["max_features", "max_samples"]

for year in YEARS:
    print(f"\n{'='*60}")
    print(f"Loading year {year}...")
    df = load_dataset(year=year, include_aod=False)
    dfx, features = prepare_features(df, "non_aod")
    print(f"  {len(dfx)} rows, {len(features)} features")

    for hp in HYPERPARAMS:
        hp_range = config.SWEEP_RANGES[hp]
        param_values = np.arange(hp_range["range"][0], hp_range["range"][1], hp_range["step"])

        for cv_name, block_col in [("Spatial", "index"), ("Temporal", "random_dateid")]:
            print(f"\n  Sweeping {hp} x {cv_name}-CV ({len(param_values)} values)...")
            t0 = time.time()

            sweep_results = []
            for i, val in enumerate(param_values):
                if hp_range["dtype"] == int:
                    val = int(val)

                params = config.DEFAULT_RF_PARAMS.copy()
                params[hp] = val
                cv = BlockCV(params)

                result = cv.perform_cv(
                    dfx, features, config.TARGET, block_col,
                    n_splits=N_SPLITS, n_iterations=N_ITER
                )

                sweep_results.append({
                    "R2_mean": result["R2_mean"],
                    "R2_std": result["R2_std"],
                    "RMSE_mean": result["RMSE_mean"],
                    "RMSE_std": result["RMSE_std"],
                    hp: val,
                })

                if (i + 1) % 5 == 0 or i == len(param_values) - 1:
                    print(f"    [{i+1}/{len(param_values)}] {hp}={val} "
                          f"R2={result['R2_mean']:.4f}")

            elapsed = time.time() - t0
            outfile = config.RESULTS_DIR / f"sweep_{year}_{hp}_{cv_name}.csv"
            pd.DataFrame(sweep_results).to_csv(outfile, index=False)
            print(f"    Saved {outfile.name} ({elapsed:.0f}s)")

print(f"\n{'='*60}")
print("All sweeps done!")
print("Output files in results/ directory:")
for f in sorted(config.RESULTS_DIR.glob("sweep_*.csv")):
    print(f"  {f.name}")
