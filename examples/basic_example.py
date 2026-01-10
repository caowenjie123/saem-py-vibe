"""
Basic Example: Python saemix Library

This example demonstrates the complete workflow of the saemix library:
1. Data loading and preparation
2. Model definition
3. Model fitting with SAEM algorithm
4. Conditional distribution estimation
5. Model comparison
6. Simulation from fitted model
7. Result export

Uses theophylline PK data from the R saemix package.
"""

import numpy as np
import pandas as pd
import os

from saemix import (
    saemix,
    saemix_data,
    saemix_model,
    saemix_control,
    conddist_saemix,
    compare_saemix,
    simulate_saemix,
    save_results,
    export_to_csv,
    set_plot_options,
    get_plot_options,
    reset_plot_options,
)


def model1cpt(psi, id, xidep):
    """One-compartment PK model with first-order absorption."""
    dose = xidep[:, 0]
    tim = xidep[:, 1]
    ka = psi[id, 0]
    V = psi[id, 1]
    CL = psi[id, 2]
    k = CL / V
    ka_safe = np.where(np.abs(ka - k) < 1e-10, ka + 1e-10, ka)
    ypred = (
        dose
        * ka_safe
        / (V * (ka_safe - k))
        * (np.exp(-k * tim) - np.exp(-ka_safe * tim))
    )
    return np.maximum(ypred, 1e-10)


print("=" * 60)
print("Python saemix Example: Theophylline PK Analysis")
print("=" * 60)


# Load theophylline data
data_path = os.path.join("saemix-main", "data", "theo.saemix.tab")
if os.path.exists(data_path):
    data = pd.read_csv(data_path, sep=" ")
    print(
        f"\nLoaded data: {len(data)} observations from {data['Id'].nunique()} subjects"
    )
else:
    print("\nUsing sample data...")
    data = pd.DataFrame(
        {
            "Id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "Dose": [320] * 12,
            "Time": [0.5, 1, 2, 4] * 3,
            "Concentration": [
                4.0,
                8.0,
                7.0,
                5.0,
                3.5,
                7.5,
                6.5,
                4.5,
                4.5,
                8.5,
                7.5,
                5.5,
            ],
        }
    )

# Create saemix data object
saemix_data_obj = saemix_data(
    name_data=data,
    name_group="Id",
    name_predictors=["Dose", "Time"],
    name_response="Concentration",
)

print(f"Number of subjects: {saemix_data_obj.n_subjects}")
print(f"Total observations: {saemix_data_obj.n_total_obs}")

# Define the model
print("\n" + "-" * 60)
print("Model Definition")
print("-" * 60)

model_full = saemix_model(
    model=model1cpt,
    description="One-compartment PK model (full random effects)",
    psi0=np.array([[1.5, 30.0, 2.0]]),
    name_modpar=["ka", "V", "CL"],
    transform_par=[1, 1, 1],
    covariance_model=np.eye(3),
    omega_init=np.diag([0.5, 0.5, 0.5]),
    error_model="constant",
)

print(f"Model: {model_full.description}")
print(f"Parameters: {model_full.name_modpar}")
print(f"Error model: {model_full.error_model}")


# Fit the model
print("\n" + "-" * 60)
print("Model Fitting")
print("-" * 60)

control = saemix_control(
    nbiter_saemix=(50, 30),
    seed=12345,
    display_progress=True,
    warnings=False,
    map=True,
    fim=True,
)

print("Running SAEM algorithm...")
result = saemix(model_full, saemix_data_obj, control)

print("\n--- Fixed Effects Estimates ---")
if result.results.fixed_effects is not None:
    for name, val in zip(model_full.name_modpar, result.results.fixed_effects):
        print(f"  {name}: {val:.4f}")

print("\n--- Variance Components (Omega diagonal) ---")
if result.results.omega is not None:
    omega_diag = np.diag(result.results.omega)
    for name, val in zip(model_full.name_modpar, omega_diag):
        print(f"  omega_{name}: {val:.4f}")


# Conditional Distribution Estimation
print("\n" + "-" * 60)
print("Conditional Distribution Estimation")
print("-" * 60)

print("Computing conditional distributions using MCMC...")
result = conddist_saemix(result, nsamp=10, max_iter=100, seed=42)

print("\n--- Conditional Shrinkage ---")
for name, shrink in zip(model_full.name_modpar, result.results.cond_shrinkage):
    print(f"  {name}: {shrink:.2%}")

print("\n--- Conditional Mean (first 3 subjects) ---")
print("  Subject | " + " | ".join(f"{name:>8}" for name in model_full.name_modpar))
print("  " + "-" * 40)
for i in range(min(3, result.results.cond_mean_phi.shape[0])):
    values = " | ".join(f"{v:8.4f}" for v in result.results.cond_mean_phi[i, :])
    print(f"  {i+1:7d} | {values}")


# Model Comparison
print("\n" + "-" * 60)
print("Model Comparison")
print("-" * 60)

model_reduced = saemix_model(
    model=model1cpt,
    description="One-compartment PK model (reduced random effects)",
    psi0=np.array([[1.5, 30.0, 2.0]]),
    name_modpar=["ka", "V", "CL"],
    transform_par=[1, 1, 1],
    covariance_model=np.diag([1, 1, 0]),
    omega_init=np.diag([0.5, 0.5, 0.01]),
    error_model="constant",
)

print("Fitting reduced model...")
control_reduced = saemix_control(
    nbiter_saemix=(50, 30), seed=12345, display_progress=False, warnings=False
)
result_reduced = saemix(model_reduced, saemix_data_obj, control_reduced)

print("\nComparing models...")
comparison = compare_saemix(
    result, result_reduced, method="is", names=["Full Model", "Reduced Model"]
)

print("\n--- Model Comparison Results ---")
print(comparison.to_string(index=False))

best_model = comparison.loc[comparison["AIC"].idxmin(), "model"]
print(f"\nBest model by AIC: {best_model}")


# Simulation
print("\n" + "-" * 60)
print("Simulation")
print("-" * 60)

print("Simulating from fitted model...")
sim_data = simulate_saemix(result, nsim=100, seed=123, predictions=True, res_var=True)

print(f"\nSimulation results: {len(sim_data)} rows")
print(f"Columns: {list(sim_data.columns)}")

print("\n--- Simulation Summary (ysim) ---")
print(f"  Mean: {sim_data['ysim'].mean():.4f}")
print(f"  Std:  {sim_data['ysim'].std():.4f}")
print(f"  Min:  {sim_data['ysim'].min():.4f}")
print(f"  Max:  {sim_data['ysim'].max():.4f}")


# Export Results
print("\n" + "-" * 60)
print("Export Results")
print("-" * 60)

output_dir = "example_results"
os.makedirs(output_dir, exist_ok=True)

print(f"Saving results to '{output_dir}/'...")
save_results(result, directory=output_dir, overwrite=True)

export_to_csv(
    result, os.path.join(output_dir, "individual_params.csv"), what="individual"
)

print("\nExported files:")
for f in os.listdir(output_dir):
    print(f"  - {f}")


# Plot Options
print("\n" + "-" * 60)
print("Plot Options")
print("-" * 60)

set_plot_options(figsize=(12, 8), dpi=150, alpha=0.7)

options = get_plot_options()
print("Current plot options:")
print(f"  Figure size: {options.figsize}")
print(f"  DPI: {options.dpi}")
print(f"  Alpha: {options.alpha}")

reset_plot_options()
print("\nPlot options reset to defaults.")

# Summary
print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
print(
    f"""
Summary:
- Fitted one-compartment PK model to {saemix_data_obj.n_subjects} subjects
- Estimated conditional distributions with MCMC
- Compared full vs reduced random effects models
- Best model: {best_model}
- Generated {sim_data['sim'].nunique()} simulation replicates
- Results saved to '{output_dir}/'
"""
)
