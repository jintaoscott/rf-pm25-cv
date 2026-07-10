"""
AOD analysis: quantify the AOD contribution under each CV scheme and AOD
treatment, all years (2015-2019), Spatial-CV and Temporal-CV, 10-fold x 10-iter.

Metrics note: full-sample rows report means over the 10 iterations (R2_mean,
R2_std); the [C] subset rows (aod_present / aod_missing) are computed from the
representative iteration's predictions, so their R2_std is empty.

Three handling modes for the AOD missingness (AOD/AOD_a missing are coded 99999,
NOT NaN, so the default dropna() keeps them -> AOD and Non-AOD share the same N):

  [A] sentinel_full : current paper method. 99999 kept. Non-AOD vs AOD on the full N.
                      -> "operational" question: does a deployed model gain from
                         AOD-where-available?
  [C] decomposition : same sentinel_full runs as [A], but the test-set R2 is split into
                      AOD-present rows (both AOD & AOD_a real) vs AOD-missing rows.
                      -> "is the AOD gain concentrated where AOD is actually observed?"
                      (reuses [A]'s predictions, no extra fitting)
  [B] clear_sky     : drop rows where AOD or AOD_a == 99999 (keep both-satellite-real).
                      Non-AOD vs AOD on this matched clear-sky subset.
                      -> "physical" question: does the AOD *observation* carry signal?

Outputs (results/):
  aod_coverage.csv      - per-year AOD coverage statistics
  aod_analysis_main.csv - per (Year, CV, Mode, Model, Subset): R2, RMSE, N

Usage:
    python server_steps/step06_aod_analysis.py            # full: 10-fold x 10-iter (paper setting)
    python server_steps/step06_aod_analysis.py --quick    # sanity check: 5-fold x 1-iter, fast
    python server_steps/step06_aod_analysis.py --iter 5   # custom iterations

Expected full runtime: ~1-3 h on a multi-core server (40 CV runs x 100 fits).
Run --quick first to confirm it works end-to-end before launching the full job.
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from src.data import load_dataset, prepare_features
from src.block_cv import BlockCV
from src import config

SENTINEL = 99999
AOD_COLS = ["AOD", "AOD_a"]
YEARS = [2015, 2016, 2017, 2018, 2019]
CV_TYPES = [("Spatial", "index"), ("Temporal", "random_dateid")]

NON_AOD_FEATURES = config.FEATURES["non_aod"]
AOD_FEATURES = config.FEATURES["aod"]


def aod_present_mask(dfx):
    """Boolean array (dfx order): True where BOTH satellites have a real (non-sentinel) AOD."""
    m = np.ones(len(dfx), dtype=bool)
    for c in AOD_COLS:
        m &= (dfx[c].values != SENTINEL)
    return m


def subset_r2(y_test, y_pred, mask):
    if mask.sum() < 10:
        return np.nan, np.nan, int(mask.sum())
    r2 = r2_score(y_test[mask], y_pred[mask])
    rmse = float(np.sqrt(mean_squared_error(y_test[mask], y_pred[mask])))
    return float(r2), rmse, int(mask.sum())


def run_one(dfx, features, block_col, n_splits, n_iter):
    cv = BlockCV(config.DEFAULT_RF_PARAMS.copy())
    return cv.perform_cv(dfx, features, config.TARGET, block_col,
                         n_splits=n_splits, n_iterations=n_iter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="5-fold x 1-iter sanity run")
    ap.add_argument("--iter", type=int, default=None, help="override n_iterations")
    ap.add_argument("--splits", type=int, default=None, help="override n_splits")
    args = ap.parse_args()

    if args.quick:
        n_splits, n_iter = 5, 1
    else:
        n_splits, n_iter = 10, 10
    if args.splits: n_splits = args.splits
    if args.iter:   n_iter = args.iter
    print(f"Config: n_splits={n_splits}, n_iter={n_iter}")

    cov_rows, main_rows = [], []

    for year in YEARS:
        print(f"\n{'='*64}\nLoading year {year} ...")
        df = load_dataset(year=year, include_aod=True)

        # --- AOD model dfx (sentinel kept) and the clear-sky subset ---
        dfx_aod, _ = prepare_features(df, "aod")          # full N, 99999 kept
        mask_full = aod_present_mask(dfx_aod)
        dfx_clear = dfx_aod[mask_full].copy()             # both-satellite-real subset

        n_full = len(dfx_aod)
        n_clear = len(dfx_clear)
        # per-column coverage
        cov = {"Year": year, "N_full": n_full, "N_both_real": n_clear,
               "pct_both_real": round(100 * n_clear / n_full, 2)}
        for c in AOD_COLS:
            real = int((dfx_aod[c].values != SENTINEL).sum())
            cov[f"pct_{c}_real"] = round(100 * real / n_full, 2)
        cov_rows.append(cov)
        print(f"  N_full={n_full}  both-AOD-real={n_clear} ({cov['pct_both_real']}%)  "
              f"AOD_real={cov['pct_AOD_real']}%  AOD_a_real={cov['pct_AOD_a_real']}%")
        pd.DataFrame(cov_rows).to_csv(config.RESULTS_DIR / "aod_coverage.csv", index=False)

        for cv_name, block_col in CV_TYPES:
            print(f"  -- {cv_name}-CV --")

            # ===== [A] sentinel_full + [C] decomposition =====
            # Run AOD-model and Non-AOD-model on the SAME dfx_aod (rows aligned to mask_full)
            for model_name, feats in [("AOD", AOD_FEATURES), ("Non-AOD", NON_AOD_FEATURES)]:
                t0 = time.time()
                res = run_one(dfx_aod, feats, block_col, n_splits, n_iter)
                y_test, y_pred = res["y_test"], res["y_pred"]
                # [A] overall
                main_rows.append({
                    "Year": year, "CV": cv_name, "Mode": "sentinel_full",
                    "Model": model_name, "Subset": "all",
                    "R2_mean": round(res["R2_mean"], 4), "R2_std": round(res["R2_std"], 4),
                    "RMSE_mean": round(res["RMSE_mean"], 2), "N": n_full,
                    "N_features": len(feats),
                })
                # [C] decomposition (representative iteration)
                for sub_name, m in [("aod_present", mask_full), ("aod_missing", ~mask_full)]:
                    r2, rmse, n = subset_r2(y_test, y_pred, m)
                    main_rows.append({
                        "Year": year, "CV": cv_name, "Mode": "sentinel_full",
                        "Model": model_name, "Subset": sub_name,
                        "R2_mean": round(r2, 4) if r2 == r2 else np.nan,
                        "R2_std": np.nan,
                        "RMSE_mean": round(rmse, 2) if rmse == rmse else np.nan, "N": n,
                        "N_features": len(feats),
                    })
                print(f"     [A/C] {model_name:8s} all R2={res['R2_mean']:.4f}  ({time.time()-t0:.0f}s)")

            # ===== [B] clear_sky (matched both-satellite-real subset) =====
            for model_name, feats in [("AOD", AOD_FEATURES), ("Non-AOD", NON_AOD_FEATURES)]:
                t0 = time.time()
                res = run_one(dfx_clear, feats, block_col, n_splits, n_iter)
                main_rows.append({
                    "Year": year, "CV": cv_name, "Mode": "clear_sky",
                    "Model": model_name, "Subset": "all",
                    "R2_mean": round(res["R2_mean"], 4), "R2_std": round(res["R2_std"], 4),
                    "RMSE_mean": round(res["RMSE_mean"], 2), "N": n_clear,
                    "N_features": len(feats),
                })
                print(f"     [B]   {model_name:8s} clear R2={res['R2_mean']:.4f}  ({time.time()-t0:.0f}s)")

            # save incrementally
            pd.DataFrame(main_rows).to_csv(
                config.RESULTS_DIR / "aod_analysis_main.csv", index=False)

    print(f"\n{'='*64}\nDONE. Wrote results/aod_coverage.csv and results/aod_analysis_main.csv")

    # tidy summary: dAOD = AOD - Non-AOD, per Year/CV/Mode/Subset
    dfm = pd.DataFrame(main_rows)
    piv = dfm.pivot_table(index=["CV", "Mode", "Subset"], columns="Model",
                          values="R2_mean", aggfunc="mean")
    if {"AOD", "Non-AOD"}.issubset(piv.columns):
        piv["dAOD"] = (piv["AOD"] - piv["Non-AOD"]).round(4)
    print("\n=== AOD vs Non-AOD R2 (averaged over years) ===")
    print(piv.to_string())


if __name__ == "__main__":
    main()
