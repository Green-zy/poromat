import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax.random as random
import joblib
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive
from poromat.config import MODEL_PATHS


def bnn_model(X, y=None):
    hidden_dim = 20

    w1 = numpyro.sample("w1", dist.Normal(0, 1).expand([X.shape[1], hidden_dim]))
    b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([hidden_dim]))

    w2 = numpyro.sample("w2", dist.Normal(0, 1).expand([hidden_dim]))
    b2 = numpyro.sample("b2", dist.Normal(0, 1))

    sigma = numpyro.sample("sigma", dist.Exponential(1.0))

    hidden = jnp.tanh(jnp.dot(X, w1) + b1)
    mean = jnp.dot(hidden, w2) + b2

    numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)


def predict_stress_curve_bnn(porosity_value, T_value, rate_value, strain_step=0.005):
    """
    Predict stress-strain curve using the BNN model.

    Parameters
    ----------
    porosity_value : float
        Porosity value.
    T_value : float
        Temperature in Kelvin.
    rate_value : float
        Strain rate.
    strain_step : float
        Step size for strain.

    Returns
    -------
    strain_range : np.ndarray
        Strain values.
    stress_median : np.ndarray
        Median stress prediction.
    stress_lower : np.ndarray
        Lower bound of 95% CI.
    stress_upper : np.ndarray
        Upper bound of 95% CI.
    """
    strain_range = np.arange(0.01, 0.25, strain_step)

    # Prepare test inputs
    test_array = np.stack([
        T_value * np.ones_like(strain_range),
        rate_value * np.ones_like(strain_range),
        strain_range,
        porosity_value * np.ones_like(strain_range)
    ], axis=1)

    # Convert test input to DataFrame to match scaler feature names
    feature_names = ["T", "strainrate", "strain", "porosity"]
    test_df = pd.DataFrame(test_array, columns=feature_names)

    scaler_X = joblib.load(MODEL_PATHS["bnn"]["scaler_X"])
    scaler_y = joblib.load(MODEL_PATHS["bnn"]["scaler_y"])
    samples = joblib.load(MODEL_PATHS["bnn"]["samples"])

    X_test = scaler_X.transform(test_df)
    X_test_jnp = jnp.array(X_test)

    predictive = Predictive(bnn_model, samples, return_sites=["obs"])
    rng_key = random.PRNGKey(1)
    preds = predictive(rng_key, X=X_test_jnp)["obs"]  # (n_samples, n_points)

    stress_median = scaler_y.inverse_transform(jnp.median(preds, axis=0).reshape(-1, 1)).flatten()
    stress_lower = scaler_y.inverse_transform(jnp.percentile(preds, 2.5, axis=0).reshape(-1, 1)).flatten()
    stress_upper = scaler_y.inverse_transform(jnp.percentile(preds, 97.5, axis=0).reshape(-1, 1)).flatten()

    # Add physical origin (0, 0)
    strain_range = np.insert(strain_range, 0, 0.0)
    stress_median = np.insert(stress_median, 0, 0.0)
    stress_lower = np.insert(stress_lower, 0, 0.0)
    stress_upper = np.insert(stress_upper, 0, 0.0)

    return strain_range, stress_median, stress_lower, stress_upper
