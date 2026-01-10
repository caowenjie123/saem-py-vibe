#!/usr/bin/env python3
"""
Test Importance Sampling with different error models.
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

# Common model parameters
common_params = {
    "model": model1cpt,
    "description": "One-compartment PK model",
    "psi0": np.array([[1.5, 30.0, 2.0]]),
    "name_modpar": ["ka", "V", "CL"],
    "transform_par": [1, 1, 1],
    "covariance_model": np.eye(3),
    "omega_init": np.diag([0.5, 0.5, 0.5]),
}

# Error models to test
error_models = [
    ("constant", "constant"),
    ("proportional", "proportional"),
    ("combined", "combined"),
    ("exponential", "exponential"),
]

control = saemix_control(
    nbiter_saemix=(30, 20),  # Reduced iterations for speed
    seed=12345,
    display_progress=False,
    warnings=False,
    map=True,
    fim=False,
    ll_is=False,
    ll_gq=False,
)

print("Testing Importance Sampling with different error models")
print("=" * 60)

for model_name, error_model in error_models:
    print(f"\n{model_name.upper()} error model:")
    print("-" * 40)

    # Create model with specific error model
    model = saemix_model(**common_params, error_model=error_model)

    # Run SAEM (quick run)
    print(f"  Running SAEM...")
    result = saemix(model, saemix_data_obj, control)

    # Compute GQ likelihood
    print(f"  Computing GQ likelihood...")
    result_gq = llgq_saemix(result)
    ll_gq = result_gq.results.ll_gq

    # Compute IS likelihood with moderate sample size
    print(f"  Computing IS likelihood...")
    result.options["nmc_is"] = 2000
    result.options["debug_is"] = False
    result_is = llis_saemix(result)
    ll_is = result_is.results.ll_is

    # Compute difference
    diff = ll_is - ll_gq
    abs_diff = abs(diff)

    print(f"    GQ: {ll_gq:.3f}")
    print(f"    IS: {ll_is:.3f}")
    print(f"    Difference (IS - GQ): {diff:.3f}")

    if abs_diff < 2.0:
        print(f"    ✓ Within tolerance (±2 log-likelihood units)")
    else:
        print(f"    ✗ Exceeds tolerance")

    # Check if results are reasonable (not catastrophic failures)
    if ll_is < -10000 or ll_is > 1000:
        print(f"    ⚠️  IS value {ll_is:.1f} seems implausible")

print("\n" + "=" * 60)
print("Error model testing complete.")
