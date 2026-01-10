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

print("respar:", result.results.respar)
print("error_model:", result.model.error_model)
print("omega:", np.diag(result.results.omega))
