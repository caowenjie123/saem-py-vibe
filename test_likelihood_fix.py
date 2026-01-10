#!/usr/bin/env python3
"""
Test the fixed IS algorithm against GQ for theophylline PK model.
"""

import sys

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import os
from saemix import saemix, saemix_data, saemix_model, saemix_control
from saemix.algorithm.likelihood import llis_saemix, llgq_saemix


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


# Load theophylline data
data_path = os.path.join("saemix-main", "data", "theo.saemix.tab")
data = pd.read_csv(data_path, sep=" ")
saemix_data_obj = saemix_data(
    name_data=data,
    name_group="Id",
    name_predictors=["Dose", "Time"],
    name_response="Concentration",
)

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

control = saemix_control(
    nbiter_saemix=(50, 30),  # Reduced iterations for speed (enough for likelihood test)
    seed=12345,
    display_progress=False,
    warnings=False,
    map=True,
    fim=True,
    ll_is=False,
    ll_gq=False,
)

print("Running SAEM algorithm...")
result = saemix(model_full, saemix_data_obj, control)
print("SAEM completed.")

# Compute GQ likelihood (reference)
print("\nComputing Gaussian Quadrature likelihood...")
result_gq = llgq_saemix(result)
ll_gq = result_gq.results.ll_gq
print(f"GQ log-likelihood: {ll_gq:.3f}")

# Compute IS likelihood with different sample sizes
print("\nComputing Importance Sampling likelihood...")

# Debug run with small sample size
print("\n--- DEBUG RUN ---")
result.options["debug_is"] = True
result.options["debug_small"] = True
result.options["nmc_is"] = 4
result_debug = llis_saemix(result)
print("--- END DEBUG ---")

if True:
    # Debug IS with moderate sample size
    result.options["debug_is"] = True
    result.options["debug_small"] = False
    result.options["nmc_is"] = 1000
    result_is = llis_saemix(result)
    ll_is = result_is.results.ll_is
    print(f"IS log-likelihood (nmc_is=1000): {ll_is:.3f}")

    # Compute difference
    diff = ll_is - ll_gq
    print(f"\nDifference (IS - GQ): {diff:.3f}")
    print(f"Absolute difference: {abs(diff):.3f}")

    # Check convergence
    if abs(diff) < 2.0:
        print("✓ IS and GQ agree within tolerance (±2 log-likelihood units).")
    else:
        print("✗ IS and GQ discrepancy exceeds tolerance.")
