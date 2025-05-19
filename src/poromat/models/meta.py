import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
import os
import learn2learn as l2l
from .meta_net import StressRegressor
from ..config import MODEL_PATHS


def get_task_data(task_id, df, k_support=10, k_query=20):
    """Extract support and query sets for a given task."""
    features = ["T", "strainrate", "strain", "porosity"]
    target = "stress"
    
    task_data = df[df["task_id"] == task_id]
    task_data = task_data.sample(frac=1).reset_index(drop=True)
    x = torch.tensor(task_data[features].values, dtype=torch.float32)
    y = torch.tensor(task_data[target].values, dtype=torch.float32).squeeze()
    return (x[:k_support], y[:k_support]), (x[k_support:k_support+k_query], y[k_support:k_support+k_query])


def predict_stress_curve_meta(porosity_value, T_value=400, rate_value=2000, 
                              strain_step=0.005, n_forward_passes=50):
    """
    Predict stress-strain curve using meta-learning (MAML) model.
    Replicates the exact same logic as predict_stress_curve_meta in the notebook.
    
    Parameters
    ----------
    porosity_value : float
        Porosity value
    T_value : float
        Temperature value
    rate_value : float
        Strain rate value
    strain_step : float
        Strain step size (default: 0.005)
    n_forward_passes : int
        Number of MC Dropout forward passes for uncertainty estimation
    
    Returns
    -------
    strain_range : np.ndarray
        Strain values
    stress_median : np.ndarray
        Median stress predictions
    stress_lower : np.ndarray
        Lower bound of uncertainty (95% CI)
    stress_upper : np.ndarray
        Upper bound of uncertainty (95% CI)
    """
    
    # Load scalers (same as in training)
    scaler_X = joblib.load(MODEL_PATHS["meta_scaler_X"])
    scaler_y = joblib.load(MODEL_PATHS["meta_scaler_y"])
    
    # Load best parameters from training results
    para_error_path = "results/para_error/meta_mae_params.csv"
    if os.path.exists(para_error_path):
        result_df = pd.read_csv(para_error_path)
        best_params = result_df.iloc[0].to_dict()
        # Convert back to proper types
        best_params['hidden_dim'] = int(best_params['hidden_dim'])
        best_params['inner_steps'] = int(best_params['inner_steps'])
        best_params['support_size'] = int(best_params['support_size'])
        best_params['query_size'] = int(best_params['query_size'])
    else:
        # Fallback to default parameters
        best_params = {
            'inner_lr': 0.035132983676407964,
            'outer_lr': 0.007332002642413114,
            'meta_batch_size': 8,
            'hidden_dim': 32,
            'inner_steps': 3,
            'num_iterations': 1000,
            'support_size': 10,
            'query_size': 5
        }
    
    # Load the preprocessed training data (needed for task_id matching and support sets)
    # This should be the same preprocessing as in train_meta.py
    data_file = "data/full_data.csv"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Training data not found at {data_file}")
    
    # Load and preprocess data exactly as in training
    df = pd.read_csv(data_file)
    df["strainrate"] = df["strainrate"].clip(lower=1e-6)
    df["strain"] = df["strain"].clip(lower=1e-6)
    
    features = ["T", "strainrate", "strain", "porosity"]
    target = "stress"
    
    # Apply the same scaling as in training
    df[features] = scaler_X.transform(df[features])
    df[target] = scaler_y.transform(df[[target]])
    
    # Create task_id exactly as in training
    df["task_id"] = df.apply(lambda row: f"{row['porosity']}_{row['T']}_{row['strainrate']}", axis=1)
    
    # Create and load the MAML model
    model = StressRegressor(hidden_dim=best_params["hidden_dim"], dropout_p=0.03)
    maml = l2l.algorithms.MAML(model, lr=best_params["inner_lr"])
    maml = joblib.load(MODEL_PATHS["meta"])
    
    # 1. Generate test data
    strain_range = np.arange(0.01, 0.25, strain_step)
    test_df = pd.DataFrame({
        "T": T_value,
        "strainrate": rate_value,
        "strain": strain_range,
        "porosity": porosity_value
    })
    X_test_scaled = scaler_X.transform(test_df)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    # 2. Get support set and train a learner
    # Create task_id with the same format as in training
    task_id = f"{porosity_value}_{T_value}_{rate_value}"
    
    # Get support set from the training data
    # Since the exact task_id might not exist, find the closest match or use the first available task
    all_tasks = df["task_id"].unique()
    if task_id in all_tasks:
        selected_task_id = task_id
    else:
        # Use first available task as fallback (same as in notebook evaluation)
        selected_task_id = all_tasks[0]
    
    (x_spt, y_spt), _ = get_task_data(selected_task_id, df, 
                                     k_support=best_params["support_size"],
                                     k_query=best_params["query_size"])
    
    learner = maml.clone()
    
    # Fine-tune the learner on support set
    for _ in range(best_params["inner_steps"]):
        spt_loss = F.mse_loss(learner(x_spt), y_spt)
        learner.adapt(spt_loss)
    
    # 3. MC Dropout Inference (Enable Dropout, repeat multiple forward passes)
    learner.train()  # Enable dropout
    preds_list = []
    
    for _ in range(n_forward_passes):
        with torch.no_grad():
            preds = learner(X_test).numpy()
            preds_rescaled = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
            preds_list.append(preds_rescaled)
    
    preds_array = np.stack(preds_list)  # shape = (n_forward_passes, n_points)
    
    # 4. Calculate mean, std for 95% CI
    stress_median = np.mean(preds_array, axis=0)
    stress_std = np.std(preds_array, axis=0)
    stress_lower = stress_median - 2 * stress_std
    stress_upper = stress_median + 2 * stress_std
    
    # 5. Insert origin (0, 0) - same as in notebook
    strain_range = np.insert(strain_range, 0, 0.0)
    stress_median = np.insert(stress_median, 0, 0.0)
    stress_lower = np.insert(stress_lower, 0, 0.0)
    stress_upper = np.insert(stress_upper, 0, 0.0)
    
    return strain_range, stress_median, stress_lower, stress_upper