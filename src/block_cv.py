"""
Block cross-validation for spatiotemporal data.
Supports spatial blocking (by station) and temporal blocking (by date).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict


class BlockCV:
    def __init__(self, model_params=None):
        self.model_params = model_params or {"random_state": 42, "n_jobs": -1}
        self.model = RandomForestRegressor(**self.model_params)

    def perform_cv(self, dfx, features, target, block_column,
                   n_splits=5, random_seed=42, n_iterations=1,
                   tuning=False, tuning_param=None,
                   tuning_range=None, tuning_step=None):
        """
        Block cross-validation with optional single-parameter sweep.

        Returns
        -------
        If tuning=False: dict with R2_mean, R2_std, RMSE_mean, RMSE_std, y_pred, y_test
        If tuning=True:  list of dicts (one per parameter value)
        """
        np.random.seed(random_seed)

        if tuning:
            if not all([tuning_param, tuning_range, tuning_step]):
                raise ValueError("tuning_param, tuning_range, tuning_step required")
            param_values = np.arange(tuning_range[0], tuning_range[1], tuning_step)
            results = []
            for val in param_values:
                self.model_params[tuning_param] = val
                self.model = RandomForestRegressor(**self.model_params)
                result = self._run_cv(dfx, features, target, block_column,
                                      n_splits, n_iterations)
                result[tuning_param] = val
                results.append(result)
            return results

        return self._run_cv(dfx, features, target, block_column,
                            n_splits, n_iterations)

    def _run_cv(self, dfx, features, target, block_column,
                n_splits, n_iterations):
        r2_scores, rmse_scores, iteration_results = [], [], []

        for _ in range(n_iterations):
            blocks = dfx[block_column].unique()
            np.random.shuffle(blocks)
            folds = np.array_split(blocks, n_splits)
            y_pred = np.full(len(dfx), np.nan)

            for test_blocks in folds:
                train = ~dfx[block_column].isin(test_blocks)
                test = dfx[block_column].isin(test_blocks)

                self.model.fit(dfx.loc[train, features], dfx.loc[train, target])
                y_pred[test.values] = self.model.predict(dfx.loc[test, features])

            r2 = r2_score(dfx[target], y_pred)
            rmse = np.sqrt(mean_squared_error(dfx[target], y_pred))
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            iteration_results.append((r2, y_pred.copy(), dfx[target].values.copy()))

        # Select the iteration closest to mean R2
        if n_iterations >= 5:
            mean_r2 = np.mean(r2_scores)
            idx = np.argmin([abs(r - mean_r2) for r, _, _ in iteration_results])
        else:
            idx = 0

        return {
            "R2_mean": np.mean(r2_scores),
            "R2_std": np.std(r2_scores),
            "RMSE_mean": np.mean(rmse_scores),
            "RMSE_std": np.std(rmse_scores),
            "y_pred": iteration_results[idx][1],
            "y_test": iteration_results[idx][2],
        }

    def perform_random_cv(self, dfx, features, target,
                          n_splits=5, random_seed=42, n_iterations=1):
        """Standard KFold random CV for comparison."""
        r2_scores, rmse_scores, iteration_results = [], [], []

        for i in range(n_iterations):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed + i)
            X, y = dfx[features], dfx[target]
            y_pred = cross_val_predict(self.model, X, y, cv=kf, n_jobs=-1)

            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            iteration_results.append((r2, y_pred.copy(), y.values.copy()))

        if n_iterations >= 5:
            mean_r2 = np.mean(r2_scores)
            idx = np.argmin([abs(r - mean_r2) for r, _, _ in iteration_results])
        else:
            idx = 0

        return {
            "R2_mean": np.mean(r2_scores),
            "R2_std": np.std(r2_scores),
            "RMSE_mean": np.mean(rmse_scores),
            "RMSE_std": np.std(rmse_scores),
            "y_pred": iteration_results[idx][1],
            "y_test": iteration_results[idx][2],
        }
