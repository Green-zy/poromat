import os
import pandas as pd
from datetime import datetime

def save_stress_strain_csv(strain, stress, filename_prefix, model_name,
                            output_dir="results/outputs",
                            porosity=None, temperature=None, strainrate=None):
    """
    Save stress-strain curve to CSV, with optional metadata in header.

    Parameters
    ----------
    strain : np.ndarray
        Strain values.
    stress : np.ndarray
        Predicted stress values.
    filename_prefix : str
        Custom prefix to include in filename.
    model_name : str
        Model used (e.g., 'lightgbm', 'bnn').
    output_dir : str
        Folder to save results.
    porosity : float or None
        Porosity value to include in metadata (optional).
    temperature : float or None
        Temperature value in degrees Celsius (optional).
    strainrate : float or None
        Strain rate in 1/s (optional).
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame({
        "strain": strain,
        "stress": stress
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{model_name}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    # Write metadata to the beginning of the CSV file
    with open(filepath, "w") as f:
        f.write(f"# model = {model_name}\n")
        if porosity is not None:
            f.write(f"# porosity = {porosity}\n")
        if temperature is not None:
            f.write(f"# temperature = {temperature} K\n")
        if strainrate is not None:
            f.write(f"# strain rate = {strainrate} 1/s\n")
        f.write("# -----------------------------\n")
        df.to_csv(f, index=False, float_format="%.4f")

    print(f"[SAVED] CSV saved to: {filepath}")


