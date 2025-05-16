from .pipeline.generate_predictions import generate_prediction
from .utils.io import save_stress_strain_csv

__all__ = ["plot", "save_csv"]


def plot(porosity, temperature, strain_rate, step=0.005, method="bnn"):
    """
    Plot the stress-strain curve using the specified model.

    Parameters
    ----------
    porosity : float
        Porosity (0–40)
    temperature : float
        Temperature in Kelvin
    strain_rate : float
        Strain rate (1/s)
    step : float
        Strain step (default: 0.005)
    method : str
        One of 'bnn', 'lightgbm', 'interpolation'
    """
    return generate_prediction(
        model_name=method,
        porosity=porosity,
        T=temperature,
        rate=strain_rate,
        strain_step=step,
        save_csv=False,
        show_plot=True,
    )


def save_csv(porosity, temperature, strain_rate, step=0.005, method="bnn", path=None):
    """
    Save predicted stress-strain data to CSV using the specified model.

    Parameters
    ----------
    porosity : float
        Porosity (0–40)
    temperature : float
        Temperature in Kelvin
    strain_rate : float
        Strain rate (1/s)
    step : float
        Strain step (default: 0.005)
    method : str
        One of 'bnn', 'lightgbm', 'interpolation'
    path : str or None
        Folder to save output CSV (default: 'results/outputs')
    """
    strain, stress = generate_prediction(
        model_name=method,
        porosity=porosity,
        T=temperature,
        rate=strain_rate,
        strain_step=step,
        save_csv=False,
        show_plot=False,
    )[:2]  # Only take (strain, stress)

    filename_prefix = f"{method}_por{porosity}_T{temperature}_rate{strain_rate}_step{step}"
    save_stress_strain_csv(strain, stress, filename_prefix=filename_prefix,
                           model_name=method, output_dir=path or "results/outputs")
