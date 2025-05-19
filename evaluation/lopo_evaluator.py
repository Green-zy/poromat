# lopo_evaluator.py

import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

class LOPOEvaluator:
    def __init__(self, model_classes, strain_grid):
        self.model_classes = model_classes  # name -> class
        self.strain_grid = strain_grid
        self.results = []

    def evaluate(self, df):
        unique_porosity = df["porosity"].unique()

        # Subset config
        subset_conditions = {
            0:   [(0, 20, 3000), (0, 200, 3000), (0, 400, 3000)],
            26:  [(26, 25, 2300), (26, 100, 3000), (26, 200, 2800)],
            36:  [(36, 25, 2000), (36, 100, 2350), (36, 300, 3000)],
        }

        for p_holdout in unique_porosity:
            df_train = df[df["porosity"] != p_holdout].reset_index(drop=True)

            # Apply test subset filter
            df_test_full = df[df["porosity"] == p_holdout]
            cond = subset_conditions.get(p_holdout, [])
            mask = df_test_full.apply(lambda row: (row["porosity"], row["T"], row["strainrate"]) in cond, axis=1)
            df_test = df_test_full[mask].reset_index(drop=True)
            
            print(f"Leave-out porosity = {p_holdout}")
            round_results = {"porosity": p_holdout}


            for name, ModelClass in self.model_classes.items():
                model = ModelClass()
                model.fit(df_train)
                y_preds = model.predict(df_test, self.strain_grid)  # (n_curves, n_points)
                y_trues = []
                for (p, T, r), group in df_test.groupby(["porosity", "T", "strainrate"]):
                    group_sorted = group.sort_values("strain")
                    y_true_curve = np.interp(self.strain_grid, group_sorted["strain"], group_sorted["stress"])
                    y_trues.append(y_true_curve)

                y_trues = np.array(y_trues)
                mae = np.mean(np.abs(y_preds - y_trues))
                round_results[name] = mae

            self.results.append(round_results)

        return pd.DataFrame(self.results)

def analyze_significance(results_df):
    print("\n Friedman test:")
    scores = [results_df[col] for col in results_df.columns if col != "porosity"]
    stat, p = friedmanchisquare(*scores)
    print(f"Friedman p = {p:.4f}")
    if p < 0.05:
        print("Significant difference detected among models.")
        print("\n Posthoc Nemenyi test:")
        nemenyi = sp.posthoc_nemenyi_friedman(results_df.drop(columns="porosity"))
        print(nemenyi.round(4))
        return nemenyi
    else:
        print("No significant difference detected.")
        return None
