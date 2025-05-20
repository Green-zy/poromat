import warnings
import os
from ..config import MODEL_PATHS
from ..models.lightgbm import predict_stress_curve_lgb
from ..models.interpolation import predict_interp
from ..models.meta import predict_stress_curve_meta
from ..utils.plot import plot_stress_curve
from ..utils.io import save_stress_strain_csv


def generate_prediction(model_name, porosity, T, rate, strain_step=0.005,
                        save_csv=False, show_plot=False, output_dir=None):
    """
    Generate stress-strain predictions using the specified model.

    Parameters
    ----------
    model_name : str
        One of 'lightgbm', 'interpolation', or 'meta'
    porosity : float
        Porosity value (0-40)
    T : float
        Temperature in degrees Celsius (recommended: 20-400)
    rate : float
        Strain rate (1/s) (recommended: 500-4500)
    strain_step : float
        Strain step size (default: 0.005)
    save_csv : bool
        Whether to save results to CSV
    show_plot : bool
        Whether to display the plot
    output_dir : str or None
        Directory to save CSV files (default: 'results/outputs')

    Returns
    -------
    strain : np.ndarray
        Strain values
    stress : np.ndarray
        Stress predictions
    stress_lower : np.ndarray or None
        Lower bound of uncertainty (only for meta model)
    stress_upper : np.ndarray or None
        Upper bound of uncertainty (only for meta model)
    """

    # Input Validation
    if porosity < 0:
        raise ValueError("porosity can not be negative")
    if porosity > 40:
        warnings.warn("recommended porosity from 0 to 40", UserWarning)

    if not (20 <= T <= 400):
        warnings.warn("recommended T from 20 to 400 Celsius degrees", UserWarning)

    if rate <= 0:
        raise ValueError("strainrate must be positive")
    if rate < 500 or rate > 4500:
        warnings.warn("recommended strainrate from 500 to 4500", UserWarning)

     # Check model file(s)
    if model_name == "meta":
        required_files = [
            MODEL_PATHS["meta"],
            MODEL_PATHS["meta_scaler_X"],
            MODEL_PATHS["meta_scaler_y"]
        ]
    elif model_name == "lightgbm":
        required_files = [MODEL_PATHS["lightgbm"]]
    elif model_name == "interpolation":
        required_files = [MODEL_PATHS["interpolation"]]
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         "Choose from 'lightgbm', 'interpolation', or 'meta'.")

    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(
                f"Required file not found: {file}\n"
                f"Run `poromat.download_all_models()` to download all required model files."
            )   

    # Model Prediction
    if model_name == "lightgbm":
        strain, stress = predict_stress_curve_lgb(porosity, T, rate, strain_step)
        stress_lower, stress_upper = None, None
        ci = False

    elif model_name == "interpolation":
        strain, stress = predict_interp(porosity, T, rate, strain_step)
        stress_lower, stress_upper = None, None
        ci = False

    elif model_name == "meta":
        strain, stress, stress_lower, stress_upper = predict_stress_curve_meta(
            porosity_value=porosity, T_value=T, rate_value=rate, strain_step=strain_step
        )
        ci = True

    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         "Choose from 'lightgbm', 'interpolation', or 'meta'.")

    title = f"{model_name.capitalize()} Model: Porosity={porosity}, T={T}°C, Rate={rate}/s"

    if show_plot:
        plot_stress_curve(
            strain, stress, stress_lower, stress_upper,
            title=title, label=f"{model_name.capitalize()} Prediction", ci=ci
        )

    if save_csv:
        filename_prefix = f"por{porosity}_T{T}_rate{rate}_step{strain_step}"
        save_stress_strain_csv(
            strain, stress, filename_prefix, model_name,
            output_dir or "results/outputs",
            porosity=porosity, temperature=T, strainrate=rate
        )

    return strain, stress, stress_lower, stress_upper
