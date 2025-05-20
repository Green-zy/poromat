from .pipeline.generate_predictions import generate_prediction
from .utils.io import save_stress_strain_csv
from .utils.download import download_model, download_all_models, download_data

__all__ = ["plot", "save_csv", "download_model", "download_all_models", "download_data"]


def plot(porosity, temperature, strain_rate, step=0.005, method="meta"):
    """
    Plot the stress-strain curve using the specified model.

    Parameters
    ----------
    porosity : float
        Porosity (0–40)
    temperature : float
        Temperature in degrees Celsius
    strain_rate : float
        Strain rate (1/s)
    step : float
        Strain step (default: 0.005)
    method : str
        One of 'meta', 'lightgbm', or 'interpolation'
    """
    generate_prediction(
        model_name=method,
        porosity=porosity,
        T=temperature,
        rate=strain_rate,
        strain_step=step,
        save_csv=False,
        show_plot=True,
    )


def save_csv(porosity, temperature, strain_rate, step=0.005, method="meta", path=None):
    """
    Save predicted stress-strain data to CSV using the specified model.

    Parameters
    ----------
    porosity : float
        Porosity (0–40)
    temperature : float
        Temperature in degrees Celsius
    strain_rate : float
        Strain rate (1/s)
    step : float
        Strain step (default: 0.005)
    method : str
        One of 'meta', 'lightgbm', or 'interpolation'
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
