#!/usr/bin/env python3
"""
Test Gaussian proposal (nu_is large) vs Student's t proposal.
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

# Create model with constant error
model = saemix_model(
    model=model1cpt,
    description="One-compartment PK model",
    psi0=np.array([[1.5, 30.0, 2.0]]),
    name_modpar=["ka", "V", "CL"],
    transform_par=[1, 1, 1],
    covariance_model=np.eye(3),
    omega_init=np.diag([0.5, 0.5, 0.5]),
    error_model="constant",
)

control = saemix_control(
    nbiter_saemix=(30, 20),
    seed=12345,
    display_progress=False,
    warnings=False,
    map=True,
    fim=False,
    ll_is=False,
    ll_gq=False,
)

print("Running SAEM to get parameter estimates...")
result = saemix(model, saemix_data_obj, control)

# Compute GQ likelihood as reference
print("\nComputing GQ likelihood (reference)...")
result_gq = llgq_saemix(result)
ll_gq = result_gq.results.ll_gq
print(f"GQ log-likelihood: {ll_gq:.3f}")

# Test different nu_is values
nu_is_values = [4, 10, 30, 100, 1000]
nmc_is = 2000

print(f"\nTesting IS with different nu_is values (nmc_is={nmc_is}):")
print("nu_is | IS LL | Difference from GQ | Within ±2?")
print("-" * 50)

for nu_is in nu_is_values:
    result.options["nu_is"] = nu_is
    result.options["nmc_is"] = nmc_is
    result.options["debug_is"] = False

    result_is = llis_saemix(result)
    ll_is = result_is.results.ll_is
    diff = ll_is - ll_gq
    abs_diff = abs(diff)

    within_tolerance = "✓" if abs_diff < 2.0 else "✗"

    print(f"{nu_is:5d} | {ll_is:7.3f} | {diff:8.3f} | {within_tolerance}")

# Also test convergence with increasing sample size for nu_is=100
print(f"\nConvergence test for nu_is=100 (Gaussian approximation):")
result.options["nu_is"] = 100
result.options["debug_is"] = False

for nmc in [500, 1000, 2000, 5000, 10000]:
    result.options["nmc_is"] = nmc
    res = llis_saemix(result)
    ll = res.results.ll_is
    diff = ll - ll_gq
    print(f"  nmc_is={nmc:6d}: LL = {ll:.3f}, diff = {diff:.3f}")

print("\nTest complete.")
