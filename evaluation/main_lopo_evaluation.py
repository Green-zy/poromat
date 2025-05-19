# main_lopo_evaluation.py

import numpy as np
import pandas as pd
import os
from datetime import datetime

from interpolation_baseline import InterpolationBaseline
from lightgbm_model import LightGBMModel
from metalearning_model import MetaLearningModel
from lopo_evaluator import LOPOEvaluator, analyze_significance

# Load data
df = pd.read_csv("data/full_data.csv")
df["strainrate"] = df["strainrate"].clip(lower=1e-6)
df["strain"] = df["strain"].clip(lower=1e-6)

# Define shared strain grid
strain_grid = np.round(np.arange(0.0, 0.255, 0.005), 4)

# Define model dictionary
models = {
    "interp": InterpolationBaseline,
    "lgbm": LightGBMModel,
    "meta": MetaLearningModel
}

# Run LOPO evaluation
evaluator = LOPOEvaluator(models, strain_grid)
results_df = evaluator.evaluate(df)

# Round to 3 decimal places
results_df = results_df.round(3)

# Run significance test and capture p-value
print("\n LOPO MAE Results:")
print(results_df)

print("\nRunning Friedman test:")
from scipy.stats import friedmanchisquare

scores = [results_df[col] for col in results_df.columns if col != "porosity"]
stat, p_value = friedmanchisquare(*scores)
print(f"Friedman p = {p_value:.4f}")
note = "Friedman p-value"

# Add the p-value as a new row to results_df
results_df_with_note = results_df.copy()
p_row = pd.Series({col: "" for col in results_df.columns}, name="Friedman_p")
p_row["meta"] = round(p_value, 3)
p_row["porosity"] = note
results_df_with_note = pd.concat([results_df_with_note, pd.DataFrame([p_row])])

# Save results
os.makedirs("results/evaluations", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"results/evaluations/lopo_mae_{timestamp}.csv"
results_df_with_note.to_csv(save_path, index=False)
print(f"\nResults (including Friedman test) saved to: {save_path}")

