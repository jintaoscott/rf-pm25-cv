"""
Run 10-fold CV multiple times with different seeds to see R² variation,
report the seed closest to the 10-run mean, and save its results
(results/fig2_representative_run.csv, used for Figure 2).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import numpy as np
from src.data import load_dataset, prepare_features
from src.block_cv import BlockCV
from src import config

print("Loading data...", flush=True)
df = load_dataset(year=2019, include_aod=False)
dfx, features = prepare_features(df, "non_aod")
print(f"  {len(dfx)} samples", flush=True)

N_RUNS = 10

print(f"\nRunning 10-fold CV × {N_RUNS} seeds...\n", flush=True)
print(f"{'Seed':>6s}  {'Random':>8s}  {'Spatial':>8s}  {'Temporal':>8s}", flush=True)
print("-" * 36, flush=True)

all_results = []
for seed in range(N_RUNS):
    cv = BlockCV({"random_state": seed, "n_jobs": -1})

    r_random = cv.perform_random_cv(dfx, features, config.TARGET, n_splits=10, random_seed=seed, n_iterations=1)
    r_spatial = cv.perform_cv(dfx, features, config.TARGET, 'index', n_splits=10, random_seed=seed, n_iterations=1)
    r_temporal = cv.perform_cv(dfx, features, config.TARGET, 'random_dateid', n_splits=10, random_seed=seed, n_iterations=1)

    print(f"{seed:6d}  {r_random['R2_mean']:8.4f}  {r_spatial['R2_mean']:8.4f}  {r_temporal['R2_mean']:8.4f}", flush=True)
    all_results.append({
        'seed': seed,
        'random': r_random['R2_mean'],
        'spatial': r_spatial['R2_mean'],
        'temporal': r_temporal['R2_mean'],
        'results': {'Random-CV': r_random, 'Spatial-CV': r_spatial, 'Temporal-CV': r_temporal}
    })

# Summary
randoms = [r['random'] for r in all_results]
spatials = [r['spatial'] for r in all_results]
temporals = [r['temporal'] for r in all_results]

print(f"\n{'':6s}  {'Random':>8s}  {'Spatial':>8s}  {'Temporal':>8s}", flush=True)
print(f"{'Mean':6s}  {np.mean(randoms):8.4f}  {np.mean(spatials):8.4f}  {np.mean(temporals):8.4f}", flush=True)
print(f"{'Std':6s}  {np.std(randoms):8.4f}  {np.std(spatials):8.4f}  {np.std(temporals):8.4f}", flush=True)
print(f"{'Min':6s}  {np.min(randoms):8.4f}  {np.min(spatials):8.4f}  {np.min(temporals):8.4f}", flush=True)
print(f"{'Max':6s}  {np.max(randoms):8.4f}  {np.max(spatials):8.4f}  {np.max(temporals):8.4f}", flush=True)

# Find the seed closest to mean for all three
mean_target = np.array([np.mean(randoms), np.mean(spatials), np.mean(temporals)])
best_seed = min(all_results, key=lambda r: abs(r['random'] - mean_target[0]) + abs(r['spatial'] - mean_target[1]) + abs(r['temporal'] - mean_target[2]))
print(f"\nBest representative seed: {best_seed['seed']} (R²: {best_seed['random']:.4f} / {best_seed['spatial']:.4f} / {best_seed['temporal']:.4f})", flush=True)

out = config.RESULTS_DIR / "fig2_representative_run.csv"
with open(out, "w", encoding="utf-8", newline="") as f:
    f.write("CV,seed,R2,RMSE,N\n")
    for cv_name in ("Random", "Spatial", "Temporal"):
        r = best_seed["results"][cv_name + "-CV"]
        f.write(f"{cv_name},{best_seed['seed']},{r['R2_mean']:.4f},{r['RMSE_mean']:.2f},{len(dfx)}\n")
print(f"Saved: {out}", flush=True)
