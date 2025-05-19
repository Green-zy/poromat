# metalearning_model.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import learn2learn as l2l
from torch import optim

class StressRegressor(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, dropout_p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class MetaLearningModel:
    def __init__(self, inner_lr=0.01, outer_lr=0.005, inner_steps=3, hidden_dim=32):
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.hidden_dim = hidden_dim

    def fit(self, df_train):
        self.features = ["T", "strainrate", "strain", "porosity"]
        self.target = "stress"

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        df_train = df_train.copy()
        df_train[self.features] = self.scaler_X.fit_transform(df_train[self.features])
        df_train[self.target] = self.scaler_y.fit_transform(df_train[[self.target]])
        df_train["task_id"] = df_train.apply(lambda row: f"{row['porosity']}_{row['T']}_{row['strainrate']}", axis=1)
        self.df_train = df_train
        self.task_ids = df_train["task_id"].unique()

        model = StressRegressor(input_dim=4, hidden_dim=self.hidden_dim)
        self.maml = l2l.algorithms.MAML(model, lr=self.inner_lr)
        self.opt = optim.Adam(self.maml.parameters(), lr=self.outer_lr)

        for iteration in range(200):
            self.opt.zero_grad()
            meta_loss = 0.0
            selected_tasks = np.random.choice(self.task_ids, size=4, replace=True)
            for task_id in selected_tasks:
                learner = self.maml.clone()
                (x_spt, y_spt), (x_qry, y_qry) = self.get_task_data(task_id, k_support=10, k_query=20)
                for _ in range(self.inner_steps):
                    spt_loss = F.mse_loss(learner(x_spt), y_spt)
                    learner.adapt(spt_loss)
                qry_loss = F.mse_loss(learner(x_qry), y_qry)
                meta_loss += qry_loss
            meta_loss /= len(selected_tasks)
            meta_loss.backward()
            self.opt.step()

    def get_task_data(self, task_id, k_support=10, k_query=20):
        task_data = self.df_train[self.df_train["task_id"] == task_id]
        task_data = task_data.sample(frac=1).reset_index(drop=True)
        x = torch.tensor(task_data[self.features].values, dtype=torch.float32)
        y = torch.tensor(task_data[self.target].values, dtype=torch.float32).squeeze()
        return (x[:k_support], y[:k_support]), (x[k_support:k_support+k_query], y[k_support:k_support+k_query])

    def predict(self, df_test, strain_grid):
        preds = []
        df_test = df_test.copy()
        grouped = df_test.groupby(["porosity", "T", "strainrate"])

        for (p, T, r), group in grouped:
            task_id = f"{p}_{T}_{r}"
            learner = self.maml.clone()
            (x_spt, y_spt), _ = self.get_task_data(task_id, k_support=10, k_query=0)

            for _ in range(self.inner_steps):
                spt_loss = F.mse_loss(learner(x_spt), y_spt)
                learner.adapt(spt_loss)

            X_grid = pd.DataFrame({
                "T": T,
                "strainrate": r,
                "strain": strain_grid,
                "porosity": p
            })
            X_scaled = self.scaler_X.transform(X_grid[self.features])
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            with torch.no_grad():
                y_pred_scaled = learner(X_tensor).numpy()
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            preds.append(y_pred)

        return np.array(preds)
