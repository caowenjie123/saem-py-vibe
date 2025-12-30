import numpy as np
import pandas as pd
from saemix import saemix, saemix_data, saemix_model, saemix_control


def model1cpt(psi, id, xidep):
    dose = xidep[:, 0]
    tim = xidep[:, 1]
    ka = psi[id, 0]
    V = psi[id, 1]
    CL = psi[id, 2]
    k = CL / V
    ypred = dose * ka / (V * (ka - k)) * (np.exp(-k * tim) - np.exp(-ka * tim))
    return ypred


data = pd.DataFrame({
    'Id': [1, 1, 1, 2, 2, 2],
    'Dose': [100, 0, 0, 100, 0, 0],
    'Time': [0, 1, 2, 0, 1, 2],
    'Concentration': [0, 5, 3, 0, 6, 4]
})

saemix_data_obj = saemix_data(
    name_data=data,
    name_group='Id',
    name_predictors=['Dose', 'Time'],
    name_response='Concentration'
)

saemix_model_obj = saemix_model(
    model=model1cpt,
    description="One-compartment model",
    psi0=np.array([[1.0, 20.0, 0.5]]),
    name_modpar=["ka", "V", "CL"],
    transform_par=[1, 1, 1],
    covariance_model=np.eye(3),
    omega_init=np.eye(3),
    error_model="constant"
)

control = saemix_control(
    seed=632545,
    save=False,
    save_graphs=False,
    print_results=False
)

result = saemix(saemix_model_obj, saemix_data_obj, control)
print(result)