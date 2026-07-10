# The Interdependence Between Cross-Validation and Hyperparameter Tuning Reveals Limits of Performance Metrics and Interpretation in Random-Forest PM2.5 Models

Analysis code, model-input data, and derived results accompanying the
manuscript:

> Gu J, Amadeh A, Mahuze R and Zhang K M, "The Interdependence Between
> Cross-Validation and Hyperparameter Tuning Reveals Limits of Performance
> Metrics and Interpretation in Random-Forest PM2.5 Models".

## What is here

| folder | contents |
|---|---|
| `data/` | six of the seven model-input tables (see `DATA.md`) |
| `src/` | data loading (national-network filter included), block cross-validation, configuration (feature lists, station subsets, sweep ranges) |
| `server_steps/` | numbered experiment steps (baselines, sweeps, Optuna, AOD analysis, SHAP, importances, multi-year sweeps) |
| `results/` | derived result tables (baseline R2/RMSE, hyperparameter sweep curves, Optuna outputs, AOD treatment analysis, importance/SHAP values, station-density ladder) |
| `tools/` | script that rebuilds the PM2.5 input table from public archives |

Six of the seven model-input tables are included under their providers'
open licences; the PM2.5 table must be obtained separately (see `DATA.md`).

## Reproducing the experiments

1. Obtain the PM2.5 observations and place the file in `data/`;
   `tools/fetch_rebuild_pm25.py` rebuilds the table from public archives
   (see `DATA.md`). The other six input files are included in the repository.
2. From the repository root, run `python server_steps/step00_verify_filter.py` first - it asserts the
   national-only station filter and per-year sample fingerprints
   (2019: N = 50,551; 141 stations reporting valid data).
3. Run the remaining steps in numeric order (see
   `server_steps/README_STEPS.md` for the step list and outputs).

Environment: Python >= 3.10, see `requirements.txt`
(results in the paper: scikit-learn 1.7.x, 128-core Linux server).

## Key results at a glance (2019, Non-AOD model)

- Random-CV / Spatial-CV / Temporal-CV R2: 0.87 / 0.85 / 0.37
- Optuna-optimal max_features: 20 of 22 (Spatial-CV) vs 3 (Temporal-CV)
- Date-ID model (date + coordinates only): Spatial-CV R2 = 0.95
- Station-density ladder: Spatial-CV falls 0.85 -> 0.41 (157 -> 20 stations)

## License

Source code is licensed under MIT (see `LICENSE`). Redistributed data
files retain the provider-specific licences listed in `DATA.md`; the MIT
licence does not apply to them. If you use this code, please cite the
paper above.
