import os
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

# Define search space
param_dist = {
    "num_leaves": randint(16, 128),
    "max_depth": randint(5, 65),
    "min_child_samples": randint(20, 50),   
    "learning_rate": uniform(0.01, 0.4),
    "colsample_bytree": uniform(0.7, 0.3), 
    "subsample": uniform(0.7, 0.3),   
    "subsample_freq": randint(1, 5),   
    "reg_alpha": uniform(0.0, 5.0),    
    "reg_lambda": uniform(0.0, 5.0),        
    "n_estimators": randint(50, 200)
}


model = lgb.LGBMRegressor(
    objective="regression",
    metric="mae",
    boosting_type="gbdt",
    verbose=-1,
    n_jobs=-1
)

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=100,
    scoring="neg_mean_absolute_error",
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

# Evaluate
best_model = search.best_estimator_
y_pred = best_model.predict(X_valid)
mae = mean_absolute_error(y_valid, y_pred)

# Save
# Optional: retrain best model on all data
final_model = lgb.LGBMRegressor(**search.best_params_)
final_model.fit(X, y)
os.makedirs("results/models", exist_ok=True)
joblib.dump(final_model, "results/models/lgb_model.pkl")
# joblib.dump(best_model, "results/models/lgb_model.pkl")

# Save metrics
result_df = pd.DataFrame([{
    "mae": round(mae, 2),
    **{
        k: round(v, 2) if isinstance(v, float) else v
        for k, v in search.best_params_.items()
    }
}])
result_df.to_csv("results/para_error/lgb_mae_params.csv", index=False)
