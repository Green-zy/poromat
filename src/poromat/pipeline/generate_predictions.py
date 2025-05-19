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
        Temperature in degrees Celsius
    rate : float
        Strain rate (1/s)
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

    # Generate plot title
    title = f"{model_name.capitalize()} Model: Porosity={porosity}, T={T}°C, Rate={rate}/s"

    # Show plot if requested
    if show_plot:
        plot_stress_curve(
            strain, stress, stress_lower, stress_upper,
            title=title, label=f"{model_name.capitalize()} Prediction", ci=ci
        )

    # Save CSV if requested
    if save_csv:
        filename_prefix = f"por{porosity}_T{T}_rate{rate}_step{strain_step}"
        save_stress_strain_csv(
            strain, stress, filename_prefix, model_name,
            output_dir or "results/outputs",
            porosity=porosity, temperature=T, strainrate=rate
        )

    return strain, stress, stress_lower, stress_upper