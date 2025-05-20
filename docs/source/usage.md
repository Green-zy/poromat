# Usage Guide

The `poromat` package provides a convenient interface for predicting stress–strain behavior of porous titanium alloys using machine learning and meta-learning models.

---

## Installation

`poromat` requires **Python 3.10**.

You can install the package via PyPI:

```bash
pip install poromat
```

After installation, you must download pretrained models and training data:

```python
import poromat

# Download models and scalers
poromat.download_all_models()

# Download training data required by the meta-learning model
poromat.download_data()
```

## Functions Overview

| Function              | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `poromat.plot()`      | Plot stress–strain curve, with optional uncertainty (only for `"meta"`).    |
| `poromat.save_csv()`  | Save predicted strain and stress values to a CSV file.                      |
| `generate_prediction()` | Predict stress–strain curve using one of the regression models.             |

---

### `poromat.plot()`

**Description:**  
Visualizes the predicted stress–strain curve for a given set of inputs.

**Parameters:**
- `porosity` (float): Porosity value in the range 0–40.
- `temperature` (float): Temperature in Celsius. Recommended: 20–400.
- `strain_rate` (float): Strain rate in 1/s. Recommended: 500–4500.
- `step` (float, optional): Strain step size (default: 0.005).
- `method` (str, optional): One of `"meta"`, `"lightgbm"`, or `"interpolation"`.

**Example:**
```python
poromat.plot(
    porosity=16,
    temperature=300,
    strain_rate=3000,
    step=0.002,
    method="meta"
)
```

---

### `poromat.save_csv()`

**Description:**  
Saves the predicted stress–strain data to a CSV file.

**Parameters:**
- `porosity` (float): Porosity value in the range 0–40.
- `temperature` (float): Temperature in Celsius.
- `strain_rate` (float): Strain rate in 1/s.
- `step` (float, optional): Strain step size (default: 0.005).
- `method` (str, optional): One of `"meta"`, `"lightgbm"`, or `"interpolation"`.
- `path` (str, optional): Folder to save the output file (default: `"results/outputs"`).

**Example:**
```python
poromat.save_csv(
    porosity=26,
    temperature=200,
    strain_rate=2500,
    step=0.01,
    method="lightgbm"
)
```

---

### `generate_prediction()`

**Description:**  
Performs prediction using the selected regression model.

**Parameters:**
- `model_name` (`str`): One of `"meta"`, `"lightgbm"`, or `"interpolation"`.
- `porosity` (`float`): Porosity value in the range 0–40.
- `T` (`float`): Temperature in Celsius.
- `rate` (`float`): Strain rate in 1/s.
- `strain_step` (`float`, optional): Strain step size (default: `0.005`).
- `save_csv` (`bool`, optional): Whether to save predictions to a CSV file (default: `False`).
- `show_plot` (`bool`, optional): Whether to display the plot (default: `False`).
- `output_dir` (`str`, optional): Output directory for CSV if `save_csv=True`.

**Returns:**
- `strain`: `np.ndarray`  
- `stress`: `np.ndarray`  
- `stress_lower`: `np.ndarray` or `None` (only for `"meta"` model)  
- `stress_upper`: `np.ndarray` or `None` (only for `"meta"` model)

**Example:**
```python
from poromat.pipeline.generate_predictions import generate_prediction

strain, stress, _, _ = generate_prediction(
    model_name="interpolation",
    porosity=36,
    T=100,
    rate=1500,
    strain_step=0.01
)
```

**Notes:**
- Only the `"meta"` model provides uncertainty estimates (`stress_lower`, `stress_upper`) via Monte Carlo dropout.
- The `"interpolation"` and `"lightgbm"` models return only deterministic predictions.
