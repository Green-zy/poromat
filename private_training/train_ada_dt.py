import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import randint, uniform

# Load dataset
df = pd.read_csv("data/full_data.csv")
features = ["porosity", "T", "strainrate", "strain"]
target = "stress"

X = df[features]
y = df[target]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define AdaBoost with default base estimator
base_tree = DecisionTreeRegressor(random_state=42)
ada = AdaBoostRegressor(estimator=base_tree, random_state=42)

# Updated param search space
param_dist = {
    "n_estimators": randint(20, 100),
    "learning_rate": uniform(0.05, 0.4),
    "estimator__max_depth": randint(25, 65)
}

search = RandomizedSearchCV(
    estimator=ada,
    param_distributions=param_dist,
    n_iter=40,
    scoring="neg_mean_absolute_error",
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
y_pred = best_model.predict(X_valid)
mae = mean_absolute_error(y_valid, y_pred)

# Save model and parameters
os.makedirs("results/models", exist_ok=True)
joblib.dump(best_model, "results/models/ada_dt_model.pkl")

result_df = pd.DataFrame([{
    "mae": round(mae, 2),
    "n_estimators": search.best_params_.get("n_estimators"),
    "learning_rate": round(search.best_params_.get("learning_rate", 0.0), 3),
    "max_depth": search.best_params_.get("estimator__max_depth")
}])

result_df.to_csv("results/para_error/ada_dt_mae_params.csv", index=False)
