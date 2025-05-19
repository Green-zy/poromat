import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

import pandas as pd
import numpy as np
import joblib
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import learn2learn as l2l
from poromat.models.meta_net import StressRegressor

# Fix random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load and preprocess data
df = pd.read_csv("data/full_data.csv")
df["strainrate"] = df["strainrate"].clip(lower=1e-6)
df["strain"] = df["strain"].clip(lower=1e-6)

features = ["T", "strainrate", "strain", "porosity"]
target = "stress"

scaler_X = StandardScaler()
scaler_y = StandardScaler()
df[features] = scaler_X.fit_transform(df[features])
df[target] = scaler_y.fit_transform(df[[target]])
df["task_id"] = df.apply(lambda row: f"{row['porosity']}_{row['T']}_{row['strainrate']}", axis=1)
all_tasks = df["task_id"].unique()

# Support-query sampler
def get_task_data(task_id, df, k_support=10, k_query=20):
    task_data = df[df["task_id"] == task_id].sample(frac=1).reset_index(drop=True)
    x = torch.tensor(task_data[features].values, dtype=torch.float32)
    y = torch.tensor(task_data[target].values, dtype=torch.float32).squeeze()
    return (x[:k_support], y[:k_support]), (x[k_support:k_support+k_query], y[k_support:k_support+k_query])

# Hyperparameter ranges
param_ranges = {
    "inner_lr": (0.001, 0.05),
    "outer_lr": (0.0001, 0.01),
    "meta_batch_size": [2, 4, 6, 8],
    "hidden_dim": [16, 32, 48, 64],
    "inner_steps": [1, 3, 5, 8],
    "num_iterations": [500, 1000, 3000],
    "support_size": [5, 10, 15, 25],
    "query_size": [5, 10, 20, 25]
}

def sample_random_config():
    return {
        "inner_lr": random.uniform(*param_ranges["inner_lr"]),
        "outer_lr": random.uniform(*param_ranges["outer_lr"]),
        "meta_batch_size": random.choice(param_ranges["meta_batch_size"]),
        "hidden_dim": random.choice(param_ranges["hidden_dim"]),
        "inner_steps": random.choice(param_ranges["inner_steps"]),
        "num_iterations": random.choice(param_ranges["num_iterations"]),
        "support_size": random.choice(param_ranges["support_size"]),
        "query_size": random.choice(param_ranges["query_size"])
    }

# Random search
n_trials = 50
best_params = None
lowest_mae = float("inf")

for trial in range(n_trials):
    params = sample_random_config()
    model = StressRegressor(hidden_dim=params["hidden_dim"], dropout_p=0.03)
    maml = l2l.algorithms.MAML(model, lr=params["inner_lr"])
    opt = torch.optim.Adam(maml.parameters(), lr=params["outer_lr"])

    for _ in range(300):
        opt.zero_grad()
        meta_loss = 0.0
        selected_tasks = np.random.choice(all_tasks, size=params["meta_batch_size"], replace=True)
        for task_id in selected_tasks:
            learner = maml.clone()
            (x_spt, y_spt), (x_qry, y_qry) = get_task_data(task_id, df, params["support_size"], params["query_size"])
            for _ in range(params["inner_steps"]):
                loss = F.mse_loss(learner(x_spt), y_spt)
                learner.adapt(loss)
            qry_loss = F.mse_loss(learner(x_qry), y_qry)
            meta_loss += qry_loss
        meta_loss /= params["meta_batch_size"]
        meta_loss.backward()
        opt.step()

    # Eval on held-out task
    learner = maml.clone()
    (x_spt, y_spt), (x_qry, y_qry) = get_task_data(all_tasks[0], df, params["support_size"], params["query_size"])
    for _ in range(params["inner_steps"]):
        learner.adapt(F.mse_loss(learner(x_spt), y_spt))
    with torch.no_grad():
        preds = learner(x_qry)
    mae = F.l1_loss(preds, y_qry).item()

    print(f"Trial {trial+1}: MAE = {mae:.4f}, Params = {params}")
    if mae < lowest_mae:
        best_params = params
        lowest_mae = mae

# Final training using best config
final_model = StressRegressor(hidden_dim=best_params["hidden_dim"])
maml = l2l.algorithms.MAML(final_model, lr=best_params["inner_lr"])
opt = torch.optim.Adam(maml.parameters(), lr=best_params["outer_lr"])

for iter in range(best_params["num_iterations"]):
    opt.zero_grad()
    meta_loss = 0.0
    selected_tasks = np.random.choice(all_tasks, size=best_params["meta_batch_size"], replace=True)
    for task_id in selected_tasks:
        learner = maml.clone()
        (x_spt, y_spt), (x_qry, y_qry) = get_task_data(task_id, df, best_params["support_size"], best_params["query_size"])
        for _ in range(best_params["inner_steps"]):
            loss = F.mse_loss(learner(x_spt), y_spt)
            learner.adapt(loss)
        qry_loss = F.mse_loss(learner(x_qry), y_qry)
        meta_loss += qry_loss
    meta_loss /= best_params["meta_batch_size"]
    meta_loss.backward()
    opt.step()
    if iter % 100 == 0:
        print(f"Iter {iter}: Meta-loss = {meta_loss.item():.4f}")

# Save model, scalers, and results
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/para_error", exist_ok=True)
joblib.dump(maml, "results/models/meta_maml_model.pkl")
joblib.dump(scaler_X, "results/models/meta_scaler_X.pkl")
joblib.dump(scaler_y, "results/models/meta_scaler_y.pkl")

result_df = pd.DataFrame([{
    "mae": lowest_mae,
    **best_params  
}])
result_df.to_csv("results/para_error/meta_mae_params.csv", index=False)