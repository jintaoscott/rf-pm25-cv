"""
STEP 03 - Single-hyperparameter sweeps, 2019, ALL FOUR MODELS (Figures 3, 4, 5a/b).

Uniform protocol across models (10-fold x 5 iterations by default), so all
Figure 4 panels use the same protocol. Output filenames:

    {Spatial,Temporal}_d12019_{hyperparameter}{suffix}.csv
    suffix: '' = Non-AOD, '_AOD' = AOD (sentinel-full), '_dll' = Date-ID,
            '_STEST1' = Distance-controlled

Run models one at a time (each is hours):
    python step03_sweeps_2019.py --model non_aod --smoke   # sanity first
    python step03_sweeps_2019.py --model non_aod
    python step03_sweeps_2019.py --model aod
    python step03_sweeps_2019.py --model date_id
    python step03_sweeps_2019.py --model distance

"""
import argparse
import time
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import numpy as np
import pandas as pd

from src import config
from src.block_cv import BlockCV
from src.data import load_dataset, prepare_features

YEAR = 2019
CV_TYPES = [("Spatial", "index"), ("Temporal", "random_dateid")]

BASE_GRIDS = {
    "min_samples_split": list(range(2, 35)),
    "max_depth": list(range(5, 36)),
    "max_samples": [round(v, 2) for v in np.arange(0.02, 1.0, 0.02)],
}

MODELS = {
    # name: (variant, suffix, include_aod, station_subset, n_features)
    "non_aod":  ("non_aod", "",        False, None, 22),
    "aod":      ("aod",     "_AOD",    True,  None, 24),
    "date_id":  ("date_id", "_dll",    False, None, 3),
    "distance": ("non_aod", "_STEST1", False, config.DISTANCE_CONTROLLED_STATIONS, 22),
}

# Default-end R2 fingerprints (national-only pipeline).
# Tuples: (expected values, tolerance, abort on mismatch).
FINGERPRINTS = {
    "non_aod":  ({"Spatial": 0.845, "Temporal": 0.365}, 0.05, True),
    "distance": ({"Spatial": 0.51,  "Temporal": 0.42},  0.08, True),
    "aod":      ({"Spatial": 0.85,  "Temporal": 0.39},  0.08, False),
    "date_id":  ({"Spatial": 0.94,  "Temporal": 0.44},  0.10, False),
}

EXPECTED_N = {"non_aod": 50551, "aod": 50551, "date_id": 50551, "distance": 11994}
DEFAULT_GRID_VALUES = {
    "max_features": None,  # set per model (n_features)
    "min_samples_split": 2,
    "max_depth": 35,
    "max_samples": 0.98,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--smoke", action="store_true",
                   help="3 values per grid, 5-fold x 1 iteration")
    p.add_argument("--iter", type=int, default=5)
    p.add_argument("--splits", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    variant, suffix, include_aod, subset, n_feat = MODELS[args.model]
    n_splits, n_iter = (5, 1) if args.smoke else (args.splits, args.iter)

    grids = dict(BASE_GRIDS)
    grids["max_features"] = list(range(1, n_feat + 1))
    if args.smoke:
        grids = {k: [v[0], v[len(v) // 2], v[-1]] for k, v in grids.items()}

    out_dir = config.RESULTS_DIR / ("figure_sweeps_2019_smoke" if args.smoke
                                    else "figure_sweeps_2019")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model={args.model} (suffix '{suffix}'), {n_splits}-fold x {n_iter} iter, "
          f"year={YEAR}", flush=True)

    df = load_dataset(year=YEAR, include_aod=include_aod)
    dfx, features = prepare_features(df, variant, station_subset=subset)

    # --- input validation ---
    print(f"N={len(dfx):,} stations={dfx['index'].nunique()} features={len(features)}")
    assert len(features) == n_feat, f"expected {n_feat} features, got {len(features)}"
    assert len(dfx) == EXPECTED_N[args.model], (
        f"N={len(dfx):,} != expected {EXPECTED_N[args.model]:,} - wrong sample - ABORT")
    assert dfx["index"].max() < 157, "non-national stations present - filter NOT active - ABORT"
    if args.model == "aod":
        pct_missing = float((dfx["AOD"] > 1000).mean() * 100)
        print(f"sentinel share (AOD missing): {pct_missing:.1f}%")
        assert pct_missing > 50, "sentinels absent - subset sample, not full - ABORT"
    if args.model == "distance":
        codes = set(dfx["Code"].unique())
        assert codes <= set(config.DISTANCE_CONTROLLED_STATIONS), "unexpected stations"

    fps, tol, hard = FINGERPRINTS[args.model]
    for hp, values in grids.items():
        for cv_name, block_col in CV_TYPES:
            t0 = time.time()
            rows = []
            out = out_dir / f"{cv_name}_d12019_{hp}{suffix}.csv"
            print(f"\nSweeping {hp} x {cv_name} ({len(values)} values)", flush=True)
            for i, value in enumerate(values, 1):
                params = dict(config.DEFAULT_RF_PARAMS)
                params[hp] = value
                r = BlockCV(params).perform_cv(
                    dfx, features, config.TARGET, block_col,
                    n_splits=n_splits, n_iterations=n_iter)
                rows.append({"R2_mean": round(r["R2_mean"], 4),
                             "R2_std": round(r["R2_std"], 4),
                             "RMSE_mean": round(r["RMSE_mean"], 2),
                             "RMSE_std": round(r["RMSE_std"], 2),
                             hp: value})
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"  [{i}/{len(values)}] {hp}={value} R2={r['R2_mean']:.4f}",
                      flush=True)
            # default-end fingerprint
            dv = n_feat if hp == "max_features" else DEFAULT_GRID_VALUES[hp]
            row = next((x for x in rows if x[hp] == dv), None)
            if row is not None:
                delta = row["R2_mean"] - fps[cv_name]
                msg = (f"  fingerprint {hp}={dv}: R2={row['R2_mean']:.4f} vs "
                       f"~{fps[cv_name]:.2f} (delta {delta:+.4f}, tol {tol})")
                print(msg, flush=True)
                if abs(delta) > tol:
                    if hard:
                        raise AssertionError("fingerprint out of tolerance - ABORT: " + msg)
                    print("  WARNING: fingerprint outside tolerance (soft check)")
            print(f"  saved {out.name} ({time.time()-t0:.0f}s)", flush=True)

    print("\nDONE:", args.model)


if __name__ == "__main__":
    main()
