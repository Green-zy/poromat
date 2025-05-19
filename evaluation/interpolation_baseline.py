# interpolation_baseline.py

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

class InterpolationBaseline:
    def __init__(self):
        self.model = None
        self.data_denoised = None

    def fit(self, df_train):
        X = df_train[["porosity", "T", "strainrate", "strain"]]
        y = df_train["stress"]

        param_dist = {
            "n_estimators": randint(20, 100),
            "learning_rate": uniform(0.05, 0.4),
            "estimator__max_depth": randint(25, 65),
        }

        base_model = AdaBoostRegressor(estimator=DecisionTreeRegressor(), random_state=42)
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=20,
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            random_state=42,
        )
        search.fit(X, y)
        self.model = search.best_estimator_

        df_train = df_train.copy()
        df_train["stress_denoised"] = self.model.predict(X)
        self.data_denoised = df_train

    def predict(self, df_test, strain_grid):
        preds = []

        for (p, T, r), group in df_test.groupby(["porosity", "T", "strainrate"]):
            p_all = sorted(self.data_denoised["porosity"].unique())
            p0, p1 = self._get_bounds(p_all, p)

            def interp_curve(p_val):
                d_p = self.data_denoised[self.data_denoised["porosity"] == p_val]
                t_all = sorted(d_p["T"].unique())
                t0, t1 = self._get_bounds(t_all, T)

                def interp_temp(t_val):
                    d_t = d_p[d_p["T"] == t_val]
                    r_all = sorted(d_t["strainrate"].unique())
                    r0, r1 = self._get_bounds(r_all, r)

                    def interp_rate(r_val):
                        d_r = d_t[d_t["strainrate"] == r_val].sort_values("strain")
                        return np.interp(strain_grid, d_r["strain"], d_r["stress_denoised"])

                    c0, c1 = interp_rate(r0), interp_rate(r1)
                    alpha = (r - r0) / (r1 - r0 + 1e-8)
                    return (1 - alpha) * c0 + alpha * c1

                c0, c1 = interp_temp(t0), interp_temp(t1)
                alpha = (T - t0) / (t1 - t0 + 1e-8)
                return (1 - alpha) * c0 + alpha * c1

            c0, c1 = interp_curve(p0), interp_curve(p1)
            alpha = (p - p0) / (p1 - p0 + 1e-8)
            final_curve = (1 - alpha) * c0 + alpha * c1
            preds.append(final_curve)

        return np.array(preds)

    def _get_bounds(self, sorted_vals, target):
        lower = max([v for v in sorted_vals if v <= target], default=sorted_vals[0])
        upper = min([v for v in sorted_vals if v >= target], default=sorted_vals[-1])
        return lower, upper
