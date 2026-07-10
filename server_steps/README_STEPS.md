# Experiment pipeline

The experiments, as numbered steps. Step 00 verifies the shared data
fingerprints (sample sizes, station counts, feature counts); steps 03, 05,
and 09 perform additional input checks, and the other steps rely on a
successful step 00 run.

Run from the repository root, with the input tables in `data/` (see `DATA.md`):

```bash
python server_steps/step00_verify_filter.py   # must print ALL FINGERPRINTS PASS
```

| step | computes | output |
|---|---|---|
| 00 | data-pipeline verification (per-year N and station counts) | - |
| 01 | baseline R2/RMSE, 4 models x 5 years x 3 CV schemes, 10-fold x 10 repetitions | `results/baseline_all_years.csv` |
| 02 | representative-seed selection for Figure 2 (10 seeds x 3 CV) | `results/fig2_representative_run.csv` |
| 03 | single-hyperparameter sweeps, 2019, per model (`--model non_aod / aod / date_id / distance`; `--smoke` for a quick check) | `results/figure_sweeps_2019/` |
| 04 | Optuna joint optimization, Non-AOD (100 trials per CV scheme) | `results/optuna_*` |
| 05 | Optuna joint optimization, AOD | `results/optuna_aod_*` |
| 06 | AOD coverage and matched AOD-vs-Non-AOD treatments by year | `results/aod_coverage.csv`, `results/aod_analysis_main.csv` |
| 07 | SHAP importances at the two Optuna optima (update `OPTUNA_WINNERS` in `src/config.py` if step 04 is rerun) | `results/shap_*.csv` |
| 08 | multi-year max_features / max_samples sweeps (Figure S1) | `results/sweep_*.csv` |
| 09 | impurity-based importances at the two optima (Figure 6) | `results/importance_*.csv` |

Reference fingerprints (2019, national network): N = 50,551 station-days,
141 stations with valid data; Non-AOD baselines approximately 0.85 (Spatial-CV)
and 0.37 (Temporal-CV); Distance-controlled subset N = 11,994.
