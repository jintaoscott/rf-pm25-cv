"""
Experiment 1: Baseline R² for all years (2015-2019), all 4 models, 3 CV methods.
10-fold × 10 iterations (full version for SI table).

Usage:
    python server_steps/step01_baseline_all_years.py

Output: results/baseline_all_years.csv
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from src.data import load_dataset, prepare_features
from src.block_cv import BlockCV
from src import config

import pandas as pd

N_SPLITS = 10
N_ITER = 10

YEARS = [2015, 2016, 2017, 2018, 2019]

MODELS = [
    ("Non-AOD",  "non_aod", None),
    ("AOD",      "aod",     None),
    ("Date-ID",  "date_id", None),
    ("Distance", "non_aod", config.DISTANCE_CONTROLLED_STATIONS),
]

CV_TYPES = [
    ("Spatial",  "index"),
    ("Temporal", "random_dateid"),
    ("Random",   None),
]

results = []
total = len(YEARS) * len(MODELS) * len(CV_TYPES)
done = 0

for year in YEARS:
    print(f"\n{'='*60}")
    print(f"Loading year {year}...")
    df = load_dataset(year=year, include_aod=True)
    print(f"  {len(df)} rows, {df['index'].nunique()} stations")

    for model_name, variant, station_subset in MODELS:
        dfx, features = prepare_features(df, variant, station_subset)

        for cv_name, block_col in CV_TYPES:
            done += 1
            t0 = time.time()
            cv = BlockCV(config.DEFAULT_RF_PARAMS.copy())

            if block_col is not None:
                result = cv.perform_cv(
                    dfx, features, config.TARGET, block_col,
                    n_splits=N_SPLITS, n_iterations=N_ITER
                )
            else:
                result = cv.perform_random_cv(
                    dfx, features, config.TARGET,
                    n_splits=N_SPLITS, n_iterations=N_ITER
                )

            elapsed = time.time() - t0
            row = {
                "Year": year,
                "Model": model_name,
                "CV": cv_name,
                "R2_mean": round(result["R2_mean"], 4),
                "R2_std": round(result["R2_std"], 4),
                "RMSE_mean": round(result["RMSE_mean"], 2),
                "RMSE_std": round(result["RMSE_std"], 2),
                "N_samples": len(dfx),
                "N_features": len(features),
            }
            results.append(row)
            print(f"  [{done}/{total}] {year} {model_name:10s} {cv_name:10s} "
                  f"R2={row['R2_mean']:.4f}+-{row['R2_std']:.4f}  "
                  f"RMSE={row['RMSE_mean']:.2f}  ({elapsed:.0f}s)")

            # Save intermediate results after each run
            pd.DataFrame(results).to_csv(
                config.RESULTS_DIR / "baseline_all_years.csv", index=False
            )

print(f"\n{'='*60}")
print(f"Done! {len(results)} results saved to results/baseline_all_years.csv")

# Print summary table
df_results = pd.DataFrame(results)
print("\n" + df_results.pivot_table(
    index=["Model", "CV"], columns="Year", values="R2_mean"
).to_string())
