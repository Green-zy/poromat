# Changelog

All notable changes to this project will be documented in this file.

---

## [0.1.2] - 2025-05-19
### Added
- `poromat.download_data()` function to download required training data (`full_data.csv`) from GitHub.
- Improved error message when data file is missing in `meta` model, guiding user to run `download_data()`.
- Documentation now includes guidance for both model and data setup.

### Changed
- Unified `download.py` to handle both model and data downloading in a consistent interface.
- Updated `__init__.py` to expose `download_data()` in public API (`__all__`).

---

## [0.1.1] - 2025-05-19
### Added
- `poromat.download_model()` and `poromat.download_all_models()` to automatically download pretrained `.pkl` files from GitHub.
- Clear error message when model files are missing, with guidance to use the download function.
- `requests` dependency for model downloading support.

### Changed
- Internal handling of model paths for improved robustness in `meta` method.

---

## [0.1.0] - 2025-05-19
### Added
- Initial release of the `poromat` package.
- Stress–strain prediction using three regression methods:
  - Meta-learning (MAML)
  - LightGBM
  - Manual interpolation with physics-informed structure
- Key public API functions:
  - `poromat.plot()`
  - `poromat.save_csv()`
- Support for predicting porous titanium alloy behavior across temperature, strain rate, and porosity inputs.
