import numpy as np
from typing import Optional
from saemix.utils import cutoff, transphi
from saemix.algorithm.map_estimation import (
    error_function,
    conditional_distribution_c,
    conditional_distribution_d,
)
from scipy.optimize import minimize


def _normalize_ytype(ytype, ntypes):
    if ytype is None:
        return None
    ytype_arr = np.asarray(ytype).astype(int)
    if ytype_arr.size == 0:
        return ytype_arr
    if ntypes > 1 and ytype_arr.min() >= 1 and ytype_arr.max() == ntypes:
        return ytype_arr - 1
    return ytype_arr


def estep(
    kiter,
    Uargs,
    Dargs,
    opt,
    mean_phi,
    varList,
    phiM,
    rng: Optional[np.random.Generator] = None,
):
    """
    E-step of the SAEM algorithm.

    Parameters
    ----------
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility. If None, creates a new default RNG.
    """
    # Use provided RNG or create a new one
    if rng is None:
        rng = np.random.default_rng()

    nb_etas = len(varList["ind_eta"])
    omega_eta = (
        varList["omega"][np.ix_(varList["ind_eta"], varList["ind_eta"])]
        if nb_etas > 0
        else np.zeros((0, 0))
    )
    domega = cutoff(np.diag(omega_eta), 1e-10) if nb_etas > 0 else np.array([])
    omega_eta = (
        omega_eta - np.diag(np.diag(omega_eta)) + np.diag(domega)
        if nb_etas > 0
        else omega_eta
    )

    try:
        chol_omega = np.linalg.cholesky(omega_eta) if nb_etas > 0 else np.zeros((0, 0))
    except np.linalg.LinAlgError:
        chol_omega = np.eye(nb_etas)

    somega = np.linalg.inv(omega_eta) if nb_etas > 0 else np.zeros((0, 0))

    mean_phiM = np.tile(mean_phi, (Uargs["nchains"], 1))
    if len(varList["ind0_eta"]) > 0:
        phiM[:, varList["ind0_eta"]] = mean_phiM[:, varList["ind0_eta"]]

    U_y, _ = compute_LLy(phiM, Dargs, varList["pres"])

    etaM = (
        phiM[:, varList["ind_eta"]] - mean_phiM[:, varList["ind_eta"]]
        if nb_etas > 0
        else np.zeros((phiM.shape[0], 0))
    )
    phiMc = phiM.copy()

    nbiter_mcmc = opt["nbiter_mcmc"]

    if nb_etas > 0 and nbiter_mcmc[0] > 0:
        for _ in range(nbiter_mcmc[0]):
            etaMc = rng.standard_normal((Dargs["NM"], nb_etas)) @ chol_omega.T
            phiMc[:, varList["ind_eta"]] = mean_phiM[:, varList["ind_eta"]] + etaMc
            Uc_y, _ = compute_LLy(phiMc, Dargs, varList["pres"])
            deltau = Uc_y - U_y
            ind = np.where(deltau < -np.log(rng.random(Dargs["NM"])))[0]
            etaM[ind, :] = etaMc[ind, :]
            U_y[ind] = Uc_y[ind]
            phiM[ind[:, None], varList["ind_eta"]] = (
                mean_phiM[ind[:, None], varList["ind_eta"]] + etaM[ind, :]
            )

    # Kernel 2: component-wise random walk (nrs2 = 1)
    if nb_etas > 0 and len(nbiter_mcmc) > 1 and nbiter_mcmc[1] > 0:
        domega2 = np.array(
            varList.get("domega2", np.ones((nb_etas, nb_etas))), dtype=float
        )
        nrs2 = 1
        for _ in range(nbiter_mcmc[1]):
            U_eta = 0.5 * np.sum(etaM * (etaM @ somega), axis=1)
            accept_counts = np.zeros(nb_etas)
            total_counts = np.zeros(nb_etas)
            for mk in range(nb_etas):
                step = domega2[mk, nrs2 - 1]
                etaMc = etaM.copy()
                etaMc[:, mk] = etaM[:, mk] + rng.standard_normal(Dargs["NM"]) * step
                phiMc[:, varList["ind_eta"]] = mean_phiM[:, varList["ind_eta"]] + etaMc
                Uc_y, _ = compute_LLy(phiMc, Dargs, varList["pres"])
                Uc_eta = 0.5 * np.sum(etaMc * (etaMc @ somega), axis=1)
                deltau = (Uc_y - U_y) + (Uc_eta - U_eta)
                ind = np.where(deltau < -np.log(rng.random(Dargs["NM"])))[0]
                if len(ind) > 0:
                    etaM[ind, :] = etaMc[ind, :]
                    U_y[ind] = Uc_y[ind]
                    U_eta[ind] = Uc_eta[ind]
                    phiM[ind[:, None], varList["ind_eta"]] = (
                        mean_phiM[ind[:, None], varList["ind_eta"]] + etaM[ind, :]
                    )
                accept_counts[mk] += len(ind)
                total_counts[mk] += Dargs["NM"]
            acc_rate = np.divide(
                accept_counts,
                total_counts,
                out=np.zeros_like(accept_counts),
                where=total_counts > 0,
            )
            domega2[:, nrs2 - 1] = domega2[:, nrs2 - 1] * (
                1 + opt["stepsize_rw"] * (acc_rate - opt["proba_mcmc"])
            )
            domega2[:, nrs2 - 1] = np.maximum(domega2[:, nrs2 - 1], 1e-8)
        varList["domega2"] = domega2

    # Kernel 3: block random walk
    if nb_etas > 0 and len(nbiter_mcmc) > 2 and nbiter_mcmc[2] > 0:
        domega2 = np.array(
            varList.get("domega2", np.ones((nb_etas, nb_etas))), dtype=float
        )
        for _ in range(nbiter_mcmc[2]):
            if nb_etas == 1:
                nrs2 = 1
            else:
                nrs2 = (kiter % (nb_etas - 1)) + 2
            block = (
                np.arange(nb_etas)
                if nrs2 >= nb_etas
                else rng.choice(nb_etas, size=nrs2, replace=False)
            )
            step = domega2[block, nrs2 - 1]
            etaMc = etaM.copy()
            etaMc[:, block] = (
                etaM[:, block] + rng.standard_normal((Dargs["NM"], nrs2)) * step
            )
            phiMc[:, varList["ind_eta"]] = mean_phiM[:, varList["ind_eta"]] + etaMc
            Uc_y, _ = compute_LLy(phiMc, Dargs, varList["pres"])
            U_eta = 0.5 * np.sum(etaM * (etaM @ somega), axis=1)
            Uc_eta = 0.5 * np.sum(etaMc * (etaMc @ somega), axis=1)
            deltau = (Uc_y - U_y) + (Uc_eta - U_eta)
            ind = np.where(deltau < -np.log(rng.random(Dargs["NM"])))[0]
            if len(ind) > 0:
                etaM[ind, :] = etaMc[ind, :]
                U_y[ind] = Uc_y[ind]
                U_eta[ind] = Uc_eta[ind]
                phiM[ind[:, None], varList["ind_eta"]] = (
                    mean_phiM[ind[:, None], varList["ind_eta"]] + etaM[ind, :]
                )
            acc_rate = len(ind) / max(Dargs["NM"], 1)
            domega2[block, nrs2 - 1] = domega2[block, nrs2 - 1] * (
                1 + opt["stepsize_rw"] * (acc_rate - opt["proba_mcmc"])
            )
            domega2[block, nrs2 - 1] = np.maximum(domega2[block, nrs2 - 1], 1e-8)
        varList["domega2"] = domega2

    # Kernel 4: MAP-based adaptive proposal
    if (
        nb_etas > 0
        and len(nbiter_mcmc) > 3
        and nbiter_mcmc[3] > 0
        and kiter < opt["nbiter_map"]
    ):
        if Dargs.get("modeltype", "structural") == "structural":
            phi_map = _compute_phi_map(mean_phi, Dargs, varList, opt)
            nchains = Uargs["nchains"]
            N = Dargs["N"]
            phi_mapM = np.tile(phi_map, (nchains, 1))
            eta_map = phi_mapM[:, varList["ind_eta"]] - mean_phiM[:, varList["ind_eta"]]
            psi_map = transphi(phi_map, Dargs["transform_par"])
            gradf0 = _grad_structural_model(psi_map, Dargs, varList)
            gradf = np.tile(gradf0, (nchains, 1))
            gradh0 = _grad_transphi(phi_map, Dargs, varList)
            gradh = np.tile(gradh0, (nchains, 1, 1))
            Gamma, chol_Gamma, inv_Gamma = _compute_gamma(
                gradf0, gradh0, varList, Dargs
            )

            U_y, _ = compute_LLy(phiM, Dargs, varList["pres"])
            U_eta = 0.5 * np.sum(etaM * (etaM @ somega), axis=1)
            for _ in range(nbiter_mcmc[3]):
                etaMc = etaM.copy()
                for i in range(Dargs["NM"]):
                    g = chol_Gamma[i % N]
                    etaMc[i, :] = eta_map[i, :] + rng.standard_normal(nb_etas) @ g.T
                phiMc = phiM.copy()
                phiMc[:, varList["ind_eta"]] = mean_phiM[:, varList["ind_eta"]] + etaMc
                Uc_y, _ = compute_LLy(phiMc, Dargs, varList["pres"])
                Uc_eta = 0.5 * np.sum(etaMc * (etaMc @ somega), axis=1)
                propc = np.zeros(Dargs["NM"])
                prop = np.zeros(Dargs["NM"])
                for i in range(Dargs["NM"]):
                    invg = inv_Gamma[i % N]
                    diffc = etaMc[i, :] - eta_map[i, :]
                    diff = etaM[i, :] - eta_map[i, :]
                    propc[i] = 0.5 * diffc @ invg @ diffc
                    prop[i] = 0.5 * diff @ invg @ diff
                deltau = (Uc_y - U_y) + (Uc_eta - U_eta) + prop - propc
                ind = np.where(deltau < -np.log(rng.random(Dargs["NM"])))[0]
                if len(ind) > 0:
                    etaM[ind, :] = etaMc[ind, :]
                    U_y[ind] = Uc_y[ind]
                    U_eta[ind] = Uc_eta[ind]
                    phiM[ind[:, None], varList["ind_eta"]] = (
                        mean_phiM[ind[:, None], varList["ind_eta"]] + etaM[ind, :]
                    )

    return {
        "varList": varList,
        "phiM": phiM,
    }


def _compute_phi_map(mean_phi, Dargs, varList, opt):
    id0 = Dargs["IdM"][: Dargs["nobs"]]
    xind = Dargs["XM"][: Dargs["nobs"]]
    yobs = Dargs["yM"][: Dargs["nobs"]]
    ytype = Dargs.get("ytype", None)
    ntypes = len(Dargs.get("error_model", ["constant"]))
    ytype_norm = _normalize_ytype(
        ytype[: Dargs["nobs"]] if ytype is not None else None, ntypes
    )
    i1_omega2 = varList["ind_eta"]
    omega_sub = varList["omega"][np.ix_(i1_omega2, i1_omega2)]
    try:
        iomega = np.linalg.inv(omega_sub)
    except np.linalg.LinAlgError:
        iomega = np.eye(len(i1_omega2))
    phi_map = mean_phi.copy()
    for i in range(Dargs["N"]):
        mask = id0 == i
        xi = xind[mask]
        yi = yobs[mask]
        idi = np.zeros(len(yi), dtype=int)
        mean_phi1 = mean_phi[i, i1_omega2]
        phii = mean_phi[i, :].copy()
        phi1 = phii[i1_omega2]
        if Dargs.get("modeltype", "structural") == "structural":

            def objective(phi1_opt):
                return conditional_distribution_c(
                    phi1_opt,
                    phii,
                    idi,
                    xi,
                    yi,
                    mean_phi1,
                    i1_omega2,
                    iomega,
                    Dargs["transform_par"],
                    Dargs["structural_model"],
                    varList["pres"],
                    Dargs["error_model"],
                    ytype_norm[mask] if ytype_norm is not None else None,
                )

        else:

            def objective(phi1_opt):
                return conditional_distribution_d(
                    phi1_opt,
                    phii,
                    idi,
                    xi,
                    yi,
                    mean_phi1,
                    i1_omega2,
                    iomega,
                    Dargs["transform_par"],
                    Dargs["structural_model"],
                )

        try:
            res = minimize(objective, phi1, method="BFGS")
            phi_map[i, i1_omega2] = res.x
        except Exception:
            phi_map[i, i1_omega2] = phi1
    return phi_map


def _grad_structural_model(psi_map, Dargs, varList):
    id0 = Dargs["IdM"][: Dargs["nobs"]]
    xind = Dargs["XM"][: Dargs["nobs"]]
    i1_omega2 = varList["ind_eta"]
    nb_etas = len(i1_omega2)
    fpred1 = Dargs["structural_model"](psi_map, id0, xind)
    gradf = np.zeros((len(id0), nb_etas))
    for j, idx in enumerate(i1_omega2):
        psi_map2 = psi_map.copy()
        step = np.where(np.abs(psi_map[:, idx]) > 1e-8, psi_map[:, idx] * 1e-3, 1e-3)
        psi_map2[:, idx] = psi_map2[:, idx] + step
        fpred2 = Dargs["structural_model"](psi_map2, id0, xind)
        for i in range(Dargs["N"]):
            mask = id0 == i
            if np.any(mask):
                gradf[mask, j] = (fpred2[mask] - fpred1[mask]) / step[i]
    return gradf


def _grad_transphi(phi_map, Dargs, varList):
    i1_omega2 = varList["ind_eta"]
    nb_etas = len(i1_omega2)
    psi_map = transphi(phi_map, Dargs["transform_par"])
    gradh = np.zeros((Dargs["N"], nb_etas, nb_etas))
    for j, idx in enumerate(i1_omega2):
        phi_map2 = phi_map.copy()
        step = np.where(np.abs(phi_map[:, idx]) > 1e-8, phi_map[:, idx] * 1e-3, 1e-3)
        phi_map2[:, idx] = phi_map2[:, idx] + step
        psi_map2 = transphi(phi_map2, Dargs["transform_par"])
        delta = (psi_map2[:, i1_omega2] - psi_map[:, i1_omega2]) / step[:, None]
        gradh[:, :, j] = delta
    return gradh


def _compute_gamma(gradf0, gradh0, varList, Dargs):
    nb_etas = len(varList["ind_eta"])
    omega_sub = varList["omega"][np.ix_(varList["ind_eta"], varList["ind_eta"])]
    try:
        inv_omega = np.linalg.inv(omega_sub)
    except np.linalg.LinAlgError:
        inv_omega = np.eye(nb_etas)
    sigma = varList["pres"][0] if len(varList.get("pres", [])) > 0 else 1.0
    Gamma = np.zeros((Dargs["N"], nb_etas, nb_etas))
    chol_Gamma = np.zeros_like(Gamma)
    inv_Gamma = np.zeros_like(Gamma)
    id0 = Dargs["IdM"][: Dargs["nobs"]]
    for i in range(Dargs["N"]):
        mask = id0 == i
        temp = gradf0[mask, :] @ gradh0[i, :, :]
        mat = (temp.T @ temp) / (sigma**2) + inv_omega
        try:
            Gamma_i = np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            Gamma_i = np.linalg.pinv(mat)
        Gamma[i, :, :] = Gamma_i
        chol_Gamma[i, :, :] = np.linalg.cholesky(Gamma_i + 1e-10 * np.eye(nb_etas))
        inv_Gamma[i, :, :] = np.linalg.inv(Gamma_i + 1e-10 * np.eye(nb_etas))
    return Gamma, chol_Gamma, inv_Gamma


def compute_LLy(phiM, Dargs, pres):
    psiM = transphi(phiM, Dargs["transform_par"])
    fpred = Dargs["structural_model"](psiM, Dargs["IdM"], Dargs["XM"])

    error_model = Dargs.get("error_model", ["constant"])
    ytype = _normalize_ytype(Dargs.get("ytype", None), len(error_model))

    if len(error_model) > 1 and ytype is not None:
        for ityp, em in enumerate(error_model):
            if em == "exponential":
                mask = ytype == ityp
                if np.any(mask):
                    fpred[mask] = np.log(cutoff(fpred[mask]))
    elif len(error_model) == 1 and error_model[0] == "exponential":
        fpred = np.log(cutoff(fpred))

    if Dargs.get("modeltype", "structural") == "structural":
        gpred = error_function(fpred, pres, error_model, ytype)
        dyf = 0.5 * ((Dargs["yM"] - fpred) / gpred) ** 2 + np.log(gpred)
    else:
        dyf = -fpred

    U = np.bincount(Dargs["IdM"], weights=dyf, minlength=Dargs["NM"])

    return U, dyf
