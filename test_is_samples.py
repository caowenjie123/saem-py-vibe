import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import os

from saemix import saemix, saemix_data, saemix_model, saemix_control
from saemix.algorithm.likelihood import llis_saemix

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

result = saemix(model_full, saemix_data_obj, control)

for nmc in [1000, 5000, 20000, 50000]:
    result.options['nmc_is'] = nmc
    result.results.ll_is = None
    result.results.aic_is = None
    result.results.bic_is = None
    try:
        llis_saemix(result)
        print(f"nmc {nmc}: ll_is = {result.results.ll_is}")
    except Exception as e:
        print(f"nmc {nmc}: error {e}")
