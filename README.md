# poromat

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Documentation Status](https://readthedocs.org/projects/poromat/badge/?version=latest)](https://poromat.readthedocs.io/en/latest/)

A Python package for modeling stress–strain behavior of porous titanium alloys under various strain rates and temperatures, based on extremely limited experimental data.

![Sample_figure](/results/outputs/sample_3d.png)

## Project Summary

This project constructs global stress–strain prediction models for porous Ti alloys using only 38 sets of uniaxial compression tests. Guided by the Zener–Arrhenius (ZA) constitutive framework, I implement three regression approaches and evaluate them using a Leave-One-Porosity-Out (LOPO) framework. Generalization performance is statistically compared via Friedman and Nemenyi tests. Meta-learning shows advantages on unseen porosity conditions.

## Dataset Structure

**Porosity 26%**
```
├── 25°C:  [1200, 2300, 3600, 5200]  
├── 100°C: [950, 2200, 3000, 4200]  
├── 200°C: [1050, 1500, 1950, 2800, 3800]  
└── 300°C: [1100, 1900, 2900, 3700]  
```

**Porosity 36%**
```
├── 25°C:  [1000, 2000, 3000]  
├── 100°C: [1300, 2050, 2350, 3400, 3700]  
└── 300°C: [1000, 2000, 3000, 4000, 4500]  
```

**Data Source:**
> Liu, Z., Ji, F., Wang, M., & Zhu, T. (2017).  
> *One-Dimensional Constitutive Model for Porous Titanium Alloy at Various Strain Rates and Temperatures.* Metals, 7(1), 24.  
> https://doi.org/10.3390/met7010024

## Regression Methods

1. **Interpolation**  
   Custom interpolator with Adaboost + Decision Tree smoothing, performing interpolation across strain rate, temperature, and porosity dimensions.

2. **LightGBM**  
   Gradient boosting regression on standardized features, trained directly on the limited dataset.

3. **Meta-learning (MAML)**  
   Two hidden layers (see [code](https://github.com/Green-zy/poromat/blob/master/src/poromat/models/meta_net.py)) trained using [learn2learn](https://learn2learn.net/) for fast adaptation to new porosity conditions.  
   Supports **credible intervals** via MC Dropout.

All the pre-trained models are saved in [Release v0.1.2-models](https://github.com/Green-zy/poromat/releases/tag/v0.1.2-models).

## Model Comparison

I conducted **three rounds of Leave-One-Porosity-Out (LOPO)** experiments, each time holding out one porosity level for testing.

### Statistical Evaluation

Model performance was compared using the **Friedman test**, followed by the **Nemenyi post-hoc test**. [Evaluation Result](https://github.com/Green-zy/poromat/tree/master/results/evaluations)

- **Null Hypothesis (H₀):** All models have equal performance (no significant difference).
- **Alternative Hypothesis (H₁):** At least one model performs significantly differently from the others.
- **Significance Level (α):** 0.05

| porosity         |   interp |   lgbm   |   meta   |
|:-----------------|:--------:|:--------:|:--------:|
| 0                | 1295.760 | 1284.390 | 955.802  |
| 26               |  184.148 |  269.047 | 100.576  |
| 36               |  261.527 |  284.586 | 177.866  |
| **Friedman p-value** |         – |       –   |  0.097   |

- No model showed statistically significant superiority (**p > 0.05**).
- However, ranking metrics consistently favored **meta-learning** on unseen porosity conditions.
- Due to only 3 porosity groups, **statistical power is limited**.

### Recommendation

Based on both performance trends and materials knowledge:

- **Meta-learning** captures more nonlinearity and adapts better to new porosity conditions.
- **Interpolation** performs similarly and is easier to interpret.

## Installation and Model File Setup

**Environment Setup:** This project requires Python 3.10. Recommend to set the environment and dependencies by **env.yml** under project root directory.

```bash
conda env create -f env.yml
conda activate poromat
```
Then, install the package by:
```bash
pip install poromat==0.1.2
```

The `.pkl` and `.csv` files are required to run predictions and are not included in the PyPI package.

You can download them using:

```python
import poromat
poromat.download_data()
poromat.download_all_models()
```

## Quick Start

```python
import poromat
poromat.download_data()
poromat.download_all_models() # You need to download the models

poromat.plot(16, 300, 3300, step=0.002, method="meta")

poromat.save_csv(16, 300, 3000, step=0.002, method="meta")

strain, stress, stress_lower, stress_upper = poromat.generate_prediction(
    model_name="meta",
    porosity=16,
    T=300,
    rate=3000,
    show_plot=True
)
```

## Key Functions

| Function                 | Description                                           |
|--------------------------|-------------------------------------------------------|
| `poromat.plot()`         | Plot stress–strain curve, with optional uncertainty (only for `"meta"` model) |
| `poromat.save_csv()`     | Save predicted strain and stress values to CSV file   |
| `generate_prediction()`  | Predict stress–strain curve using one of the regression models (`meta`, `lightgbm`, or `interpolation`) |
| `poromat.download_data()`  | Download training data required for meta-learning adaptation   |
| `poromat.download_all_models()`  | Download pre-trained models |

---

## Purpose

Designed for materials scientists studying porous titanium alloys.  
This tool enables fast and flexible access to mechanical response predictions under varying testing conditions, helping reduce the need for costly and time-consuming mechanical experiments.

---

## Developer

**Yun Zhou (Robbie)**  
Background in Mechanical Engineering and Applied Data Science.  
Email: robbiezhou1@gmail.com
GitHub: [@Green-zy](https://github.com/Green-zy)
