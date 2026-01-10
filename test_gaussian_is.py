import sys

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import os
from scipy.stats import t, norm

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

# Test Gaussian IS vs Student's t IS
print("Testing different proposal distributions for IS")
print("=" * 60)

# Extract components
model = result.model
data = result.data
res = result.results
opts = result.options

# Common setup
i1_omega2 = model.indx_omega
nphi1 = len(i1_omega2)
yobs = data.data[data.name_response].values
xind = data.data[data.name_predictors].values
index = data.data["index"].values
ytype = data.data["ytype"].values if "ytype" in data.data.columns else None
ytype_norm = _normalize_ytype(ytype, len(model.error_model))

cond_mean_phi = np.asarray(res.cond_mean_phi)
cond_var_phi = np.asarray(res.cond_var_phi)
mean_phi = np.asarray(res.mean_phi)
Omega = res.omega
pres = res.respar

omega_sub = Omega[np.ix_(i1_omega2, i1_omega2)]
IOmega = np.linalg.inv(omega_sub)

cond_var_phi1 = cutoff(cond_var_phi[:, i1_omega2])
mean_phi1 = mean_phi[:, i1_omega2]

# Test parameters
nmc_is = 5000
MM = 100
KM = max(1, int(np.round(nmc_is / MM)))

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


def run_is_with_proposal(proposal_type="t", df=4):
    meana = np.zeros(data.n_subjects)
    LL = np.zeros(KM)

    for km in range(1, KM + 1):
        if proposal_type == "t":
            r = t.rvs(df=df, size=(data.n_subjects * MM, nphi1))
            log_proposal = t.logpdf(r, df=df)
        else:  # gaussian
            r = np.random.randn(data.n_subjects * MM, nphi1)
            log_proposal = norm.logpdf(r)

        phiM1 = mtild_phiM1 + stild_phiM1 * r
        dphiM = phiM1 - mean_phiM1
        d2 = -0.5 * (np.sum(dphiM * (dphiM @ IOmega), axis=1) + c2)
        e2 = d2.reshape(data.n_subjects, MM)

        pitild_phi1 = np.sum(log_proposal, axis=1)
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

        # Use log-sum-exp for numerical stability
        max_sume = np.max(sume, axis=1, keepdims=True)
        sume_stable = sume - max_sume
        exp_sume = np.exp(sume_stable)
        newa = np.mean(exp_sume, axis=1) * np.exp(max_sume[:, 0])

        meana = meana + (newa - meana) / km
        LL[km - 1] = np.sum(np.log(cutoff(meana))) + log_const

    return LL[-1], LL


# Run tests
print("\n1. Student's t proposal (df=4) - current method:")
ll_t4, ll_hist_t4 = run_is_with_proposal("t", df=4)
print(f"   Final LL: {ll_t4:.3f}")
print(f"   Last 10 iterations: {ll_hist_t4[-10:]}")

print("\n2. Student's t proposal (df=10) - lighter tails:")
ll_t10, ll_hist_t10 = run_is_with_proposal("t", df=10)
print(f"   Final LL: {ll_t10:.3f}")
print(f"   Last 10 iterations: {ll_hist_t10[-10:]}")

print("\n3. Gaussian proposal:")
ll_gauss, ll_hist_gauss = run_is_with_proposal("gaussian")
print(f"   Final LL: {ll_gauss:.3f}")
print(f"   Last 10 iterations: {ll_hist_gauss[-10:]}")

print("\n4. Student's t proposal (df=100) - nearly Gaussian:")
ll_t100, ll_hist_t100 = run_is_with_proposal("t", df=100)
print(f"   Final LL: {ll_t100:.3f}")
print(f"   Last 10 iterations: {ll_hist_t100[-10:]}")

print(f"\nComparison with GQ likelihood: {res.ll_gq:.3f}")
print(f"Difference from GQ:")
print(f"  t(df=4): {ll_t4 - res.ll_gq:.3f}")
print(f"  t(df=10): {ll_t10 - res.ll_gq:.3f}")
print(f"  Gaussian: {ll_gauss - res.ll_gq:.3f}")
print(f"  t(df=100): {ll_t100 - res.ll_gq:.3f}")

# Check convergence
print(f"\nConvergence (last - first of last 10):")
print(f"  t(df=4): {ll_hist_t4[-1] - ll_hist_t4[-10]:.3f}")
print(f"  t(df=10): {ll_hist_t10[-1] - ll_hist_t10[-10]:.3f}")
print(f"  Gaussian: {ll_hist_gauss[-1] - ll_hist_gauss[-10]:.3f}")
print(f"  t(df=100): {ll_hist_t100[-1] - ll_hist_t100[-10]:.3f}")
