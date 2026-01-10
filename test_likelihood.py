import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import os

from saemix import saemix, saemix_data, saemix_model, saemix_control

def model1cpt(psi, id, xidep):
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

# Load data
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
    nbiter_saemix=(50, 30),
    seed=12345,
    display_progress=False,
    warnings=False,
    map=True,
    fim=True,
)

print("Running SAEM...")
result = saemix(model_full, saemix_data_obj, control)

print("Result ll_is:", result.results.ll_is)
print("Result ll_gq:", result.results.ll_gq)

# Compute IS likelihood
from saemix.algorithm.likelihood import llis_saemix, llgq_saemix
try:
    llis_saemix(result)
    print("IS likelihood computed:", result.results.ll_is)
except Exception as e:
    print("IS failed:", e)
    import traceback
    traceback.print_exc()

try:
    llgq_saemix(result)
    print("GQ likelihood computed:", result.results.ll_gq)
except Exception as e:
    print("GQ failed:", e)
    import traceback
    traceback.print_exc()
