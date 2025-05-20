# Changelog

All notable changes to this project will be documented in this file.

---

## [0.1.1] - 2024-07-03
### Added
- `poromat.download_meta_model()` function to automatically download pretrained meta model files (`.pkl`) from GitHub.
- Clear error message when meta model files are missing, with guidance to use the download function.
- `requests` dependency for model downloading support.

### Changed
- Internal handling of model paths for improved robustness in `meta` method.

---

## [0.1.0] - 2024-07-01
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

---
