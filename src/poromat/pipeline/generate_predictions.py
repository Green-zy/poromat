import os
import numpy as np
from poromat.config import MODEL_PATHS, DEFAULT_STRAIN_STEP
from poromat.models.bnn import predict_stress_curve_bnn
from poromat.models.lightgbm import predict_stress_curve_lgb
from poromat.models.interpolation import predict_interp
from poromat.utils.io import save_stress_strain_csv
from poromat.utils.plot import plot_stress_curve


def generate_prediction(
    model_name,
    porosity,
    T,
    rate,
    strain_step=DEFAULT_STRAIN_STEP,
    save_csv=False,
    show_plot=True,
):
    """
    Unified interface to generate stress-strain prediction using different models.

    Parameters
    ----------
    model_name : str
        One of 'bnn', 'lightgbm', or 'interpolation'.
    porosity : float
        Porosity value.
    T : float
        Temperature in Kelvin.
    rate : float
        Strain rate (1/s).
    strain_step : float
        Step size for strain axis (default: from config).
    save_csv : bool
        Whether to save the prediction as a CSV file.
    show_plot : bool
        Whether to display the prediction plot.
    """
    model_name = model_name.lower()
    if model_name not in {"bnn", "lightgbm", "interpolation"}:
        raise ValueError(f"Unknown model: {model_name}")

    # Format output filename prefix
    filename_prefix = f"por{porosity}_T{T}_rate{rate}"

    # Bayesian Neural Network
    if model_name == "bnn":
        strain, stress_median, stress_lower, stress_upper = predict_stress_curve_bnn(
            porosity_value=porosity,
            T_value=T,
            rate_value=rate,
            strain_step=strain_step
        )
        if show_plot:
            plot_stress_curve(
                strain, stress_median,
                stress_lower=stress_lower,
                stress_upper=stress_upper,
                title=f"BNN Prediction (Porosity={porosity}, T={T}, Rate={rate})",
                label="BNN Median",
                ci=True
            )
        if save_csv:
            save_stress_strain_csv(
                strain, stress_median,
                filename_prefix=filename_prefix,
                model_name="bnn"
            )

    # LightGBM
    elif model_name == "lightgbm":
        strain, stress = predict_stress_curve_lgb(
            porosity=porosity,
            T=T,
            rate=rate,
            strain_step=strain_step
        )
        if show_plot:
            plot_stress_curve(
                strain, stress,
                title=f"LGBM Prediction (Porosity={porosity}, T={T}, Rate={rate})",
                label="LightGBM"
            )
        if save_csv:
            save_stress_strain_csv(
                strain, stress,
                filename_prefix=filename_prefix,
                model_name="lightgbm"
            )

    # Interpolation-based Method
    elif model_name == "interpolation":
        strain, stress = predict_interp(
            porosity=porosity,
            T=T,
            rate=rate,
            strain_step=strain_step
        )
        if show_plot:
            plot_stress_curve(
                strain, stress,
                title=f"Interpolation (Porosity={porosity}, T={T}, Rate={rate})",
                label="Interpolation"
            )
        if save_csv:
            save_stress_strain_csv(
                strain, stress,
                filename_prefix=filename_prefix,
                model_name="interpolation"
            )
