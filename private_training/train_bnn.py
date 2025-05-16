import os
import pandas as pd
import numpy as np
import joblib
import jax.numpy as jnp
import jax.random as random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import numpyro.distributions as dist
from numpyro import sample
from numpyro.infer import MCMC, NUTS, Predictive

# Load and preprocess data
df = pd.read_csv("data/full_data.csv")
df["strainrate"] = df["strainrate"].clip(lower=1e-6)
df["strain"] = df["strain"].clip(lower=1e-6)

features = ["T", "strainrate", "strain", "porosity"]
target = "stress"

scaler_X = StandardScaler()
scaler_y = StandardScaler()

df[features] = scaler_X.fit_transform(df[features])
df[target] = scaler_y.fit_transform(df[[target]])

X = jnp.array(df[features].to_numpy())
y = jnp.array(df[target].to_numpy())

# BNN model
def bnn_model(X, y=None):
    hidden_dim = 32
    w1 = sample("w1", dist.Normal(0, 1).expand([X.shape[1], hidden_dim]))
    b1 = sample("b1", dist.Normal(0, 1).expand([hidden_dim]))
    w2 = sample("w2", dist.Normal(0, 1).expand([hidden_dim]))
    b2 = sample("b2", dist.Normal(0, 1))
    sigma = sample("sigma", dist.Exponential(1.0))

    hidden = jnp.tanh(jnp.dot(X, w1) + b1)
    mean = jnp.dot(hidden, w2) + b2

    sample("obs", dist.Normal(mean, sigma), obs=y)

# Run MCMC
rng_key = random.PRNGKey(0)
nuts_kernel = NUTS(bnn_model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1500)
mcmc.run(rng_key, X=X, y=y)

predictive = Predictive(bnn_model, mcmc.get_samples(), return_sites=["obs"])
preds = predictive(rng_key, X=X)["obs"]

# Median prediction and inverse transform
y_pred_median = jnp.median(preds, axis=0)
y_pred_real = scaler_y.inverse_transform(np.array(y_pred_median).reshape(-1, 1)).flatten()
y_real = scaler_y.inverse_transform(np.array(y).reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_real, y_pred_real)

# Save posterior + scalers
os.makedirs("results/models", exist_ok=True)
joblib.dump(mcmc.get_samples(), "results/models/bnn_samples.pkl")
joblib.dump(scaler_X, "results/models/bnn_scaler_X.pkl")
joblib.dump(scaler_y, "results/models/bnn_scaler_y.pkl")

# Save evaluation
result_df = pd.DataFrame([{
    "mae": round(mae, 2),
    "model": "BNN (NumPyro)",
    "samples": len(mcmc.get_samples()["w1"])  # number of posterior samples
}])
result_df.to_csv("results/para_error/bnn_mae_params.csv", index=False)