import sys

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import os
from scipy.stats import t

from saemix import saemix, saemix_data, saemix_model, saemix_control
from saemix.algorithm.likelihood import (
    _log_const_exponential,
    _normalize_ytype,
    _compute_npar_est,
)
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

# Extract components for debugging
model = result.model
data = result.data
res = result.results
opts = result.options

# Algorithm parameters
nmc_is = int(opts.get("nmc_is", 5000))
nu_is = opts.get("nu_is", 4)
MM = 100
KM = max(1, int(np.round(nmc_is / MM)))

# Data
i1_omega2 = model.indx_omega
nphi1 = len(i1_omega2)
yobs = data.data[data.name_response].values
xind = data.data[data.name_predictors].values
index = data.data["index"].values
ytype = data.data["ytype"].values if "ytype" in data.data.columns else None
ytype_norm = _normalize_ytype(ytype, len(model.error_model))

# Model parameters
cond_mean_phi = np.asarray(res.cond_mean_phi)
cond_var_phi = np.asarray(res.cond_var_phi)
mean_phi = np.asarray(res.mean_phi)
Omega = res.omega
pres = res.respar

omega_sub = Omega[np.ix_(i1_omega2, i1_omega2)]
try:
    IOmega = np.linalg.inv(omega_sub)
except np.linalg.LinAlgError:
    IOmega = np.eye(nphi1)

cond_var_phi1 = cutoff(cond_var_phi[:, i1_omega2])
mean_phi1 = mean_phi[:, i1_omega2]

mean_phiM1 = np.repeat(mean_phi1, MM, axis=0)
mtild_phiM1 = np.repeat(cond_mean_phi[:, i1_omega2], MM, axis=0)
stild_phiM1 = np.repeat(np.sqrt(cond_var_phi1), MM, axis=0)
phiM = np.repeat(cond_mean_phi, MM, axis=0)

IdM = np.tile(index, MM) + np.repeat(np.arange(MM) * data.n_subjects, len(index))
yM = np.tile(yobs, MM)
XM = np.tile(xind, (MM, 1))
ytypeM = np.tile(ytype_norm, MM) if ytype_norm is not None else None

log_const = _log_const_exponential(yobs, ytype_norm, model.error_model)
c2 = np.log(np.linalg.det(omega_sub)) + nphi1 * np.log(2 * np.pi)
c1 = np.log(2 * np.pi)

meana = np.zeros(data.n_subjects)
LL = np.zeros(KM)

print(f"Debug IS Algorithm")
print(f"==================")
print(f"n_subjects: {data.n_subjects}")
print(f"nphi1 (random effects): {nphi1}")
print(f"MM: {MM}, KM: {KM}, total samples: {MM * KM}")
print(f"nu_is: {nu_is}")
print(
    f"cond_var_phi1 range: [{np.min(cond_var_phi1):.3e}, {np.max(cond_var_phi1):.3e}]"
)
print(f"stild_phiM1 range: [{np.min(stild_phiM1):.3e}, {np.max(stild_phiM1):.3e}]")

# Test one iteration
km = 1
r = t.rvs(df=nu_is, size=(data.n_subjects * MM, nphi1))
phiM1 = mtild_phiM1 + stild_phiM1 * r
dphiM = phiM1 - mean_phiM1

# e2 term
d2 = -0.5 * (np.sum(dphiM * (dphiM @ IOmega), axis=1) + c2)
e2 = d2.reshape(data.n_subjects, MM)

# e3 term (proposal density)
log_tpdf = t.logpdf(r, df=nu_is)
pitild_phi1 = np.sum(log_tpdf, axis=1)
e3 = pitild_phi1.reshape(data.n_subjects, MM) - np.repeat(
    0.5 * np.sum(np.log(cond_var_phi1), axis=1)[:, None], MM, axis=1
)

# e1 term (likelihood)
phiM[:, i1_omega2] = phiM1
psiM = transphi(phiM, model.transform_par)
f = model.model(psiM, IdM, XM)

if model.modeltype == "structural":
    if len(model.error_model) > 1 and ytypeM is not None:
        for ityp, em in enumerate(model.error_model):
            if em == "exponential":
                mask = ytypeM == ityp
                if np.any(mask):
                    f[mask] = np.log(cutoff(f[mask]))
    elif len(model.error_model) == 1 and model.error_model[0] == "exponential":
        f = np.log(cutoff(f))
    g = error_function(f, pres, model.error_model, ytypeM)
    dyf = -0.5 * ((yM - f) / g) ** 2 - np.log(g) - 0.5 * c1
else:
    dyf = f

e1 = np.bincount(IdM, weights=dyf, minlength=data.n_subjects * MM).reshape(
    data.n_subjects, MM
)

print(f"\nTerm ranges for first subject (MM={MM} samples):")
print(f"e1 (likelihood): [{np.min(e1[0]):.3f}, {np.max(e1[0]):.3f}]")
print(f"e2 (prior): [{np.min(e2[0]):.3f}, {np.max(e2[0]):.3f}]")
print(f"e3 (proposal): [{np.min(e3[0]):.3f}, {np.max(e3[0]):.3f}]")

sume = e1 + e2 - e3
print(f"sume: [{np.min(sume[0]):.3f}, {np.max(sume[0]):.3f}]")
print(f"exp(sume): [{np.min(np.exp(sume[0])):.3e}, {np.max(np.exp(sume[0])):.3e}]")

# Check for extreme values
print(f"\nExtreme value check:")
print(f"Number of -inf in sume: {np.sum(~np.isfinite(sume))}")
print(f"Number of exp(sume) < 1e-100: {np.sum(np.exp(sume) < 1e-100)}")
print(f"Number of exp(sume) > 1e100: {np.sum(np.exp(sume) > 1e100)}")

# Test multiple iterations
print(f"\nTesting convergence over iterations:")
test_LL = []
for km in range(1, min(KM, 10) + 1):
    r = t.rvs(df=nu_is, size=(data.n_subjects * MM, nphi1))
    phiM1 = mtild_phiM1 + stild_phiM1 * r
    dphiM = phiM1 - mean_phiM1

    d2 = -0.5 * (np.sum(dphiM * (dphiM @ IOmega), axis=1) + c2)
    e2 = d2.reshape(data.n_subjects, MM)

    log_tpdf = t.logpdf(r, df=nu_is)
    pitild_phi1 = np.sum(log_tpdf, axis=1)
    e3 = pitild_phi1.reshape(data.n_subjects, MM) - np.repeat(
        0.5 * np.sum(np.log(cond_var_phi1), axis=1)[:, None], MM, axis=1
    )

    phiM[:, i1_omega2] = phiM1
    psiM = transphi(phiM, model.transform_par)
    f = model.model(psiM, IdM, XM)

    if model.modeltype == "structural":
        g = error_function(f, pres, model.error_model, ytypeM)
        dyf = -0.5 * ((yM - f) / g) ** 2 - np.log(g) - 0.5 * c1
    else:
        dyf = f

    e1 = np.bincount(IdM, weights=dyf, minlength=data.n_subjects * MM).reshape(
        data.n_subjects, MM
    )

    sume = e1 + e2 - e3
    newa = np.mean(np.exp(sume), axis=1)

    if km == 1:
        meana = newa
    else:
        meana = meana + (newa - meana) / km

    LL_km = np.sum(np.log(cutoff(meana))) + log_const
    test_LL.append(LL_km)
    print(f"  km={km}: LL = {LL_km:.3f}")

print(f"\nLL values: {test_LL}")
print(f"Range: {np.max(test_LL) - np.min(test_LL):.3f}")
