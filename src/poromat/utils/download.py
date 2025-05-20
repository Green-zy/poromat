import os
import requests

_MODEL_BASE_URL = "https://raw.githubusercontent.com/Green-zy/poromat/master/results/models/"
_DATA_BASE_URL = "https://raw.githubusercontent.com/Green-zy/poromat/master/data/"

_MODEL_FILES = {
    "lightgbm": "lgb_model.pkl",
    "interpolation": "ada_dt_model.pkl",
    "meta": "meta_maml_model.pkl",
    "meta_scaler_X": "meta_scaler_X.pkl",
    "meta_scaler_y": "meta_scaler_y.pkl"
}

_DATA_FILES = ["full_data.csv"]

def download_model(model_name):
    """
    Download a specific model file (or scaler) by name.
    """
    if model_name not in _MODEL_FILES:
        raise ValueError(f"Unknown model name: {model_name}")

    filename = _MODEL_FILES[model_name]
    url = _MODEL_BASE_URL + filename
    local_path = os.path.join("results/models", filename)
    os.makedirs("results/models", exist_ok=True)

    print(f"Downloading {filename}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
        print(f"Saved to {local_path}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

def download_all_models():
    """
    Download all required model and scaler files.
    """
    for model_name in _MODEL_FILES:
        download_model(model_name)

def download_data():
    """
    Download training data required by the meta model.
    """
    os.makedirs("data", exist_ok=True)
    for fname in _DATA_FILES:
        url = _DATA_BASE_URL + fname
        local_path = os.path.join("data", fname)
        print(f"Downloading data file: {fname}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"Saved to {local_path}")
        except Exception as e:
            print(f"Failed to download {fname}: {e}")