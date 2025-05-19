# lightgbm_model.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

class LightGBMModel:
    def __init__(self):
        self.model = None
        self.features = ["porosity", "T", "strainrate", "strain"]
        self.target = "stress"

    def fit(self, df_train):
        X = df_train[self.features]
        y = df_train[self.target]

        param_dist = {
            "num_leaves": randint(16, 128),
            "max_depth": randint(5, 65),
            "min_data_in_leaf": randint(20, 50),
            "learning_rate": uniform(0.01, 0.4),
            "feature_fraction": uniform(0.7, 0.3),
            "bagging_fraction": uniform(0.7, 0.3),
            "bagging_freq": randint(1, 5),
            "lambda_l1": uniform(0.0, 5.0),
            "lambda_l2": uniform(0.0, 5.0),
            "n_estimators": randint(50, 200)
        }

        fixed_params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "verbose": -1,
            "n_jobs": -1
        }

        model = lgb.LGBMRegressor(**fixed_params)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=40,
            scoring="neg_mean_absolute_error",
            cv=3,
            random_state=42,
            n_jobs=-1
        )
        search.fit(X, y)
        self.model = search.best_estimator_

    def predict(self, df_test, strain_grid):
        preds = []
        grouped = df_test.groupby(["porosity", "T", "strainrate"])
        for (p, T, r), _ in grouped:
            X_pred = pd.DataFrame({
                "porosity": p,
                "T": T,
                "strainrate": r,
                "strain": strain_grid
            })
            y_pred = self.model.predict(X_pred)
            preds.append(y_pred)
        return np.array(preds)
