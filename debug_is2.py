import sys

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import os
from scipy.stats import t

from saemix import saemix, saemix_data, saemix_model, saemix_control
from saemix.algorithm.likelihood import _log_const_exponential, _normalize_ytype
from saemix.utils import cutoff, transphi
from saemix.algorithm.map_estimation import error_function


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

result = saemix(model_full, saemix_data_obj, control)

# Extract components
model = result.model
data = result.data
res = result.results

# Algorithm test with small sample to analyze
np.random.seed(123)
MM = 5  # Small sample for analysis
nu_is = 4

i1_omega2 = model.indx_omega
nphi1 = len(i1_omega2)
yobs = data.data[data.name_response].values
xind = data.data[data.name_predictors].values
index = data.data["index"].values

cond_mean_phi = np.asarray(res.cond_mean_phi)
cond_var_phi = np.asarray(res.cond_var_phi)
mean_phi = np.asarray(res.mean_phi)
Omega = res.omega
pres = res.respar

omega_sub = Omega[np.ix_(i1_omega2, i1_omega2)]
IOmega = np.linalg.inv(omega_sub)

cond_var_phi1 = cutoff(cond_var_phi[:, i1_omega2])
mean_phi1 = mean_phi[:, i1_omega2]
mtild_phi1 = cond_mean_phi[:, i1_omega2]
stild_phi1 = np.sqrt(cond_var_phi1)

# Focus on first subject
isub = 0
print(f"Analyzing subject {isub + 1}")
print(f"cond_mean_phi: {mtild_phi1[isub]}")
print(f"cond_var_phi: {cond_var_phi1[isub]}")
print(f"stild_phi: {stild_phi1[isub]}")

# Generate samples
r = t.rvs(df=nu_is, size=(MM, nphi1))
phi1 = mtild_phi1[isub] + stild_phi1[isub] * r

print(f"\nSamples r (t variates):")
print(r)
print(f"\nTransformed phi1:")
print(phi1)

# Transform to psi (original scale)
phi_full = np.repeat(mean_phi[isub : isub + 1], MM, axis=0)
phi_full[:, i1_omega2] = phi1
psi = transphi(phi_full, model.transform_par)

print(f"\nTransformed psi (original scale):")
print(psi)

# Evaluate model
idi = np.zeros(len(yobs[index == isub]), dtype=int)
xi = xind[index == isub]
yi = yobs[index == isub]

print(f"\nSubject data: n_obs={len(yi)}, dose={xi[:, 0]}, time={xi[:, 1]}, conc={yi}")

f_all = []
for j in range(MM):
    psi_j = psi[j : j + 1]
    f = model.model(psi_j, idi, xi)
    f_all.append(f)
    print(f"\nSample {j}:")
    print(f"  psi: {psi_j[0]}")
    print(f"  predictions: {f}")
    print(f"  obs: {yi}")
    print(f"  residuals: {yi - f}")
    print(f"  MSE: {np.mean((yi - f) ** 2):.3f}")

# Compute weights for these samples
c2 = np.log(np.linalg.det(omega_sub)) + nphi1 * np.log(2 * np.pi)
c1 = np.log(2 * np.pi)

weights = []
for j in range(MM):
    # Prior term
    dphi = phi1[j] - mean_phi1[isub]
    e2 = -0.5 * (np.sum(dphi * (dphi @ IOmega)) + c2)

    # Proposal term
    log_tpdf = np.sum(t.logpdf(r[j], df=nu_is))
    e3 = log_tpdf - 0.5 * np.sum(np.log(cond_var_phi1[isub]))

    # Likelihood term
    psi_j = psi[j : j + 1]
    f = model.model(psi_j, idi, xi)
    g = error_function(f, pres, model.error_model, None)
    dyf = -0.5 * ((yi - f) / g) ** 2 - np.log(g) - 0.5 * c1
    e1 = np.sum(dyf)

    sume = e1 + e2 - e3
    weight = np.exp(sume)

    weights.append((j, e1, e2, e3, sume, weight))

    print(f"\nSample {j} weight components:")
    print(f"  e1 (likelihood): {e1:.3f}")
    print(f"  e2 (prior): {e2:.3f}")
    print(f"  e3 (proposal): {e3:.3f}")
    print(f"  sume: {sume:.3f}")
    print(f"  weight: {weight:.3e}")

print(f"\nSummary:")
print(f"Mean weight: {np.mean([w[5] for w in weights]):.3e}")
print(f"Std weight: {np.std([w[5] for w in weights]):.3e}")
print(
    f"Max/min weight ratio: {np.max([w[5] for w in weights]) / np.min([w[5] for w in weights]):.3e}"
)

# Check model predictions for extreme values
print(f"\nChecking model function numerically:")
test_psi = np.array([[1.5, 30.0, 2.0]])  # Typical values
f_test = model.model(test_psi, np.array([0]), np.array([[320, 0.5]]))
print(f"Test psi={test_psi[0]}, dose=320, time=0.5 -> f={f_test[0]:.3f}")

# Extreme parameter test
extreme_psi = np.array([[0.001, 0.001, 0.001]])  # Very small
f_extreme = model.model(extreme_psi, np.array([0]), np.array([[320, 0.5]]))
print(f"Extreme psi={extreme_psi[0]}, dose=320, time=0.5 -> f={f_extreme[0]:.3e}")
