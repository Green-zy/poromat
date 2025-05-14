import numpy as np
import pandas as pd
import joblib
from poromat.config import MODEL_PATHS


def predict_stress_curve_lgb(porosity, T, rate, strain_step=0.005):
    model = joblib.load(MODEL_PATHS["lightgbm"])
    strain_range = np.arange(0, 0.25, strain_step)

    test_df = pd.DataFrame({
        "porosity": porosity,
        "T": T,
        "strainrate": rate,
        "strain": strain_range
    })

    stress_pred = model.predict(test_df)
    return strain_range, stress_pred
