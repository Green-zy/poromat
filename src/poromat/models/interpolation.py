import numpy as np
import pandas as pd
import joblib

# Predefined interpolation anchors
layer1 = np.array([0, 26, 36])  # porosity levels

layer2 = np.array([
    [20, 200, 400],
    [25, 100, 200, 300],
    [25, 100, 300]
], dtype=object)

layer3 = np.array([
    [[500, 1000, 3000, 5000], [500, 3000], [500, 3000]],
    [[1200, 2300, 3600, 5200], [950, 2200, 3000, 4200],
     [1050, 1500, 1950, 2800, 3800], [1100, 1900, 2900, 3700]],
    [[1000, 2000, 3000], [1300, 2050, 2350, 3400, 3700],
     [1000, 2000, 3000, 4000, 4500]]
], dtype=object)

def find_bounds(target, candidates):
    candidates = np.array(candidates)
    if target <= candidates[0]:
        return candidates[0], candidates[0]
    if target >= candidates[-1]:
        return candidates[-1], candidates[-1]
    for i in range(len(candidates) - 1):
        if candidates[i] <= target <= candidates[i + 1]:
            return candidates[i], candidates[i + 1]
    raise ValueError(f"Cannot find bounds for {target} in {candidates}")

def predict_interp(porosity, T, rate, strain_step=0.001, model_path="results/models/ada_dt_model.pkl"):
    """
    Predict stress-strain curve using interpolation over AdaBoost predictions.

    Parameters
    ----------
    porosity : float
        Porosity value to interpolate between anchor points.
    T : float
        Temperature value.
    rate : float
        Strain rate value.
    strain_step : float
        Spacing of strain points (default: 0.001)
    model_path : str
        Path to trained AdaBoostRegressor model.

    Returns
    -------
    strain_range : np.ndarray
        Array of strain values.
    interpolated_curve : np.ndarray
        Interpolated stress predictions.
    """
    strain_range = np.arange(0, 0.25, strain_step)
    model = joblib.load(model_path)

    # Step 1: porosity bounds
    p0, p1 = find_bounds(porosity, layer1)

    # Step 2: temperature bounds
    t0_p0, t1_p0 = find_bounds(T, layer2[np.where(layer1 == p0)[0][0]])
    t0_p1, t1_p1 = find_bounds(T, layer2[np.where(layer1 == p1)[0][0]])

    # Step 3: strainrate bounds
    def get_r_bounds(p, t):
        i = np.where(layer1 == p)[0][0]
        j = np.where(layer2[i] == t)[0][0]
        return find_bounds(rate, layer3[i][j])

    r0_p0t0, r1_p0t0 = get_r_bounds(p0, t0_p0)
    r0_p0t1, r1_p0t1 = get_r_bounds(p0, t1_p0)
    r0_p1t0, r1_p1t0 = get_r_bounds(p1, t0_p1)
    r0_p1t1, r1_p1t1 = get_r_bounds(p1, t1_p1)

    def predict(p, t, r):
        X = pd.DataFrame({
            "porosity": p,
            "T": t,
            "strainrate": r,
            "strain": strain_range
        })
        return model.predict(X)

    def interp(c0, c1, alpha):
        return (1 - alpha) * c0 + alpha * c1

    # Interpolation logic
    c_p0_t0 = interp(predict(p0, t0_p0, r0_p0t0), predict(p0, t0_p0, r1_p0t0), (rate - r0_p0t0) / (r1_p0t0 - r0_p0t0 + 1e-8))
    c_p0_t1 = interp(predict(p0, t1_p0, r0_p0t1), predict(p0, t1_p0, r1_p0t1), (rate - r0_p0t1) / (r1_p0t1 - r0_p0t1 + 1e-8))
    c_p1_t0 = interp(predict(p1, t0_p1, r0_p1t0), predict(p1, t0_p1, r1_p1t0), (rate - r0_p1t0) / (r1_p1t0 - r0_p1t0 + 1e-8))
    c_p1_t1 = interp(predict(p1, t1_p1, r0_p1t1), predict(p1, t1_p1, r1_p1t1), (rate - r0_p1t1) / (r1_p1t1 - r0_p1t1 + 1e-8))

    alpha_t_p0 = (T - t0_p0) / (t1_p0 - t0_p0 + 1e-8)
    alpha_t_p1 = (T - t0_p1) / (t1_p1 - t0_p1 + 1e-8)
    c_p0 = interp(c_p0_t0, c_p0_t1, alpha_t_p0)
    c_p1 = interp(c_p1_t0, c_p1_t1, alpha_t_p1)

    alpha_p = (porosity - p0) / (p1 - p0 + 1e-8)
    final_curve = interp(c_p0, c_p1, alpha_p)

    return strain_range, final_curve
