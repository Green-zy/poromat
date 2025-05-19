"""
Generate stress-strain curves for different temperatures and strain rates using the ZA constitutive model.
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ZA strain division: before and after yield
def ZA_STRAIN_DIVISION(T, strain_ratio, yield_limit):
    elastic_moduli = 195100
    yield_strain = yield_limit / elastic_moduli
    strain_before_yield = np.arange(0, yield_strain, 0.0005)
    strain_after_yield = np.arange(yield_strain, 0.4, 0.0005)
    return strain_before_yield, strain_after_yield

# Full strain
def ZA_STRAIN(T, strain_ratio, yield_limit):
    before, after = ZA_STRAIN_DIVISION(T, strain_ratio, yield_limit)
    return np.append(before, after)

# ZA constitutive model
def ZA_CONS_MODEL(T, strain_ratio, yield_limit):
    C1, C2, C3, C4 = 167.4473, 212.567, 0.0238, 0.0029
    C5, C6, n = 6505.6, 1005.2, 1.1435
    B1, B2, B3 = 1.149, -4.576e-4, -1.241e-8

    before, after = ZA_STRAIN_DIVISION(T, strain_ratio, yield_limit)
    stress_after = (
        C1 + C2 * math.exp(-C3 * T + C4 * T * math.log(strain_ratio))
        + (C5 * after**n + C6) * (B1 + B2 * T + B3 * T**2)
    )
    stress_before = stress_after[0] / after[0] * before
    return np.append(stress_before, stress_after)

# Generalized function for curve plotting and data creation
def generate_curve(ax, T_C, strainrate, yieldlimit, porosity=0):
    T_K = T_C + 273.15
    strain = ZA_STRAIN(T_K, strainrate, yieldlimit)
    stress = ZA_CONS_MODEL(T_K, strainrate, yieldlimit)

    strain = np.round(strain, 4)

    ax.plot(strain, stress, color='orange')
    ax.set_xlabel('Strain')
    ax.set_ylabel('Stress')
    ax.set_title(f'T={T_C}℃, Strain rate={strainrate}/s')
    
    n = len(strain)
    data = pd.DataFrame({
        'stress': stress,
        'strain': strain,
        'T': [T_C] * n,
        'strainrate': [strainrate] * n,
        'porosity': [porosity] * n
    })
    return data

# Set figure
fig, axes = plt.subplots(2, 4, figsize=(20, 7))
axes = axes.flatten()

# Define test conditions
conditions = [
    (20, 500, 1184.6),
    (200, 500, 1159.6),
    (400, 500, 954.4),
    (20, 1000, 1292.4),
    (20, 3000, 1398.1),
    (200, 3000, 1364.1),
    (400, 3000, 1149.5),
    (20, 5000, 1432.3)
]

# Generate all data
all_data = []
for i, (T_C, strainrate, yieldlimit) in enumerate(conditions):
    data = generate_curve(axes[i], T_C, strainrate, yieldlimit)
    all_data.append(data)

plt.tight_layout()
plt.show()

# Save combined data
data_supplement = pd.concat(all_data, ignore_index=True)
data_supplement.to_csv("data/raw/dense.csv", index=False)
print("Saved combined data to dense.csv")
