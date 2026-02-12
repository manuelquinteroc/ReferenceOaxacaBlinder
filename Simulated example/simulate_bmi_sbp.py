"""
simulate_bmi_sbp.py

Replication of Section 4: Simulated Healthcare Example from:

“Do covariates explain why these groups differ? The choice of reference group
can reverse conclusions in the Oaxaca–Blinder decomposition.”

This script simulates two groups:
    - H (higher-resource)
    - K (lower-resource)

with:
    X = BMI
    Y = SBP (systolic blood pressure)

Data-generating process (Equation 5 in the paper):
    Y = α_g + β_g X + ε

It:
1) Simulates BMI distributions for each group
2) Generates SBP using group-specific linear models
3) Fits OLS in each group
4) Computes Oaxaca–Blinder components under H and K references
5) Saves:
       - data/bmi_sbp_data.csv   (raw simulated sample)
       - data/bmi_sbp_ob.csv     (fitted OB quantities)
6) Produces the “population (true) lines” plot:
       - Figures/bmi_sbp_comparison_population.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
np.random.seed(20250815)
N = 5000
sigma = 5.0  # noise SD in SBP

os.makedirs("data", exist_ok=True)
os.makedirs("Figures", exist_ok=True)

# -----------------------------------------------------------------------------
# 1) Simulate BMI distributions
# -----------------------------------------------------------------------------
# H: higher-resource; K: lower-resource
bmi_H = np.random.normal(25, 4, N)
bmi_K = np.random.normal(27, 4, N)

# -----------------------------------------------------------------------------
# 2) True (population) model parameters (used for the population-lines plot)
# -----------------------------------------------------------------------------
alpha_H_pop = 110.4
beta_H_pop  = 1.0
alpha_K_pop = 100.0
beta_K_pop  = 1.4

# -----------------------------------------------------------------------------
# 3) Generate SBP with noise
# -----------------------------------------------------------------------------
sbp_H = alpha_H_pop + beta_H_pop * bmi_H + np.random.normal(0, sigma, N)
sbp_K = alpha_K_pop + beta_K_pop * bmi_K + np.random.normal(0, sigma, N)

# -----------------------------------------------------------------------------
# 4) Fit regressions (OLS within each group)
# -----------------------------------------------------------------------------
def fit(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef[0], coef[1]

alpha_H_hat, beta_H_hat = fit(sbp_H, bmi_H)
alpha_K_hat, beta_K_hat = fit(sbp_K, bmi_K)

mu_H_hat = float(np.mean(bmi_H))
mu_K_hat = float(np.mean(bmi_K))

dmu_hat = mu_H_hat - mu_K_hat
db_hat  = beta_H_hat - beta_K_hat
da_hat  = alpha_H_hat - alpha_K_hat

# Oaxaca–Blinder components (two-fold, both references)
E_H = dmu_hat * beta_H_hat
U_H = mu_K_hat * db_hat + da_hat

E_K = dmu_hat * beta_K_hat
U_K = mu_H_hat * db_hat + da_hat

delta_sbp = float(np.mean(sbp_H) - np.mean(sbp_K))
sign_flip = (U_H * U_K) < 0

# -----------------------------------------------------------------------------
# 5) Save results
# -----------------------------------------------------------------------------
# Raw simulated dataset
df_data = pd.DataFrame({
    "BMI":   np.concatenate([bmi_H, bmi_K]),
    "SBP":   np.concatenate([sbp_H, sbp_K]),
    "group": ["H"] * N + ["K"] * N,
})
df_data.to_csv("data/bmi_sbp_data.csv", index=False)

# Fitted OB quantities table
df_ob = pd.DataFrame([
    ["mu_H", mu_H_hat],
    ["mu_K", mu_K_hat],
    ["delta_mu", dmu_hat],
    ["alpha_H", alpha_H_hat],
    ["beta_H", beta_H_hat],
    ["alpha_K", alpha_K_hat],
    ["beta_K", beta_K_hat],
    ["E_H", E_H],
    ["U_H", U_H],
    ["E_K", E_K],
    ["U_K", U_K],
    ["delta_SBP", delta_sbp],
    ["sign_flip", int(sign_flip)],
], columns=["quantity", "value"])
df_ob.to_csv("data/bmi_sbp_ob.csv", index=False)

# -----------------------------------------------------------------------------
# 6) Plot: Population betas with fitted values
# -----------------------------------------------------------------------------
group_H = df_data[df_data["group"] == "H"]
group_K = df_data[df_data["group"] == "K"]

# Data range with padding (same behavior as before)
bmi_min = float(df_data["BMI"].min() - 0.5)
bmi_max = float(df_data["BMI"].max() + 0.5)
sbp_min = float(df_data["SBP"].min() - 5)
sbp_max = float(df_data["SBP"].max() + 5)

x_fit = np.linspace(bmi_min, bmi_max, 200)
y_H_pop = alpha_H_pop + beta_H_pop * x_fit
y_K_pop = alpha_K_pop + beta_K_pop * x_fit

# Color-blind friendly palette (keep your mapping)
color_H = "#ff7f0e"  # H = orange
color_K = "#1f77b4"  # K = blue

# Legend handles: show opaque markers even though scatter is transparent
legend_handles = [
    Line2D([], [], marker="o", linestyle="None",
           markersize=10, markerfacecolor=color_H, markeredgecolor="none",
           alpha=1.0, label="Group H data"),
    Line2D([], [], marker="o", linestyle="None",
           markersize=10, markerfacecolor=color_K, markeredgecolor="none",
           alpha=1.0, label="Group K data"),
    Line2D([], [], color=color_H, linewidth=2.5,
           label=f"Group H (β={beta_H_pop:.2f})"),
    Line2D([], [], color=color_K, linewidth=2.5,
           label=f"Group K (β={beta_K_pop:.2f})"),
]

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(group_H["BMI"], group_H["SBP"], alpha=0.25, color=color_H, s=20, edgecolor="none")
ax.scatter(group_K["BMI"], group_K["SBP"], alpha=0.25, color=color_K, s=20, edgecolor="none")

ax.plot(x_fit, y_H_pop, color=color_H, linewidth=2.5)
ax.plot(x_fit, y_K_pop, color=color_K, linewidth=2.5)

ax.set_xlabel("BMI (kg/m²)", fontsize=20)
ax.set_ylabel("SBP (mmHg)", fontsize=20)
ax.set_xlim(bmi_min, bmi_max)
ax.set_ylim(sbp_min, sbp_max)

ax.grid(True, alpha=0.2)
ax.legend(handles=legend_handles, fontsize=20, loc="upper left", frameon=False)
ax.tick_params(axis="both", which="major", labelsize=20)

plt.tight_layout()
plt.savefig("Figures/bmi_sbp_comparison_population.pdf", format="pdf",
            bbox_inches="tight", dpi=300)
plt.show()
