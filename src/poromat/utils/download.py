import os
import requests

def download_meta_model():
    """
    Download pretrained meta model and scaler files to results/models/
    """
    base_url = "https://raw.githubusercontent.com/Green-zy/poromat/master/results/models/"
    files = ["meta_model.pkl", "meta_scaler_X.pkl", "meta_scaler_y.pkl"]

    os.makedirs("results/models", exist_ok=True)

    for fname in files:
        url = base_url + fname
        local_path = os.path.join("results/models", fname)
        print(f"Downloading {fname}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            print(f"ailed to download {fname}: {e}")
            continue

    print("All downloads completed.")
