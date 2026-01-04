import numpy as np
from saemix.algorithm.estep import estep
from saemix.algorithm.mstep import mstep
from saemix.utils import transphi


def run_saem(Dargs, Uargs, varList, opt, mean_phi, phiM, structural_model):
    nbiter_tot = opt["nbiter_saemix"][0] + opt["nbiter_saemix"][1]

    if phiM is None:
        phiM = np.tile(mean_phi, (Uargs["nchains"], 1))

    suffStat = {
        "statphi1": np.zeros((Dargs["N"], len(varList["ind_eta"]))),
        "statphi2": np.zeros((len(varList["ind_eta"]), len(varList["ind_eta"]))),
        "statphi3": np.zeros((Dargs["N"], Uargs["nb_parameters"])),
        "statrese": 0.0,
    }

    # Initialize iteration history storage
    nb_parameters = Uargs["nb_parameters"]
    n_omega = len(varList["ind_eta"])
    n_respar = len(varList.get("pres", []))

    # parpop: population parameters at each iteration (fixed effects only)
    parpop = np.zeros((nbiter_tot, nb_parameters))

    # allpar: all parameters at each iteration (fixed + omega diagonal + respar)
    n_total_params = nb_parameters + n_omega + n_respar
    allpar = np.zeros((nbiter_tot, n_total_params))

    for kiter in range(1, nbiter_tot + 1):
        xmcmc = estep(kiter, Uargs, Dargs, opt, mean_phi, varList, phiM)
        varList = xmcmc["varList"]
        phiM = xmcmc["phiM"]

        if opt["stepsize"][kiter - 1] > 0:
            xmstep = mstep(
                kiter,
                Uargs,
                Dargs,
                opt,
                structural_model,
                phiM,
                varList,
                suffStat,
                mean_phi,
            )
            varList = xmstep["varList"]
            suffStat = xmstep["suffStat"]
            mean_phi = xmstep["mean_phi"]

        # Record iteration history
        # Store population parameters (first row of mean_phi)
        parpop[kiter - 1, :] = mean_phi[0, :] if mean_phi.ndim > 1 else mean_phi

        # Store all parameters: fixed effects + omega diagonal + respar
        idx = 0
        allpar[kiter - 1, idx : idx + nb_parameters] = (
            mean_phi[0, :] if mean_phi.ndim > 1 else mean_phi
        )
        idx += nb_parameters

        # Add omega diagonal elements
        if n_omega > 0:
            omega_diag = np.diag(varList["omega"])[varList["ind_eta"]]
            allpar[kiter - 1, idx : idx + n_omega] = omega_diag
            idx += n_omega

        # Add residual error parameters
        if n_respar > 0:
            allpar[kiter - 1, idx : idx + n_respar] = varList.get("pres", [])[:n_respar]

        if kiter % opt.get("nbdisplay", 100) == 0:
            if opt.get("display_progress", False):
                print(f"Iteration {kiter}/{nbiter_tot}")

    nchains = Uargs["nchains"]
    N = Dargs["N"]
    phi_chain = phiM.reshape((nchains, N, nb_parameters))
    phi = phi_chain.mean(axis=0)

    cond_mean_phi = phi.copy()
    ind_eta = varList["ind_eta"]
    if len(ind_eta) > 0:
        cond_mean_phi[:, ind_eta] = suffStat["statphi1"]

    cond_var_phi = np.zeros_like(cond_mean_phi)
    if len(ind_eta) > 0:
        cond_var_phi[:, ind_eta] = (
            suffStat["statphi3"][:, ind_eta] - cond_mean_phi[:, ind_eta] ** 2
        )

    cond_mean_psi = transphi(cond_mean_phi, Dargs["transform_par"])

    mean_phiM = np.tile(mean_phi, (nchains, 1))
    eta = phiM - mean_phiM
    eta_chain = eta.reshape((nchains, N, nb_parameters))
    cond_mean_eta = eta_chain.mean(axis=0)
    if len(ind_eta) == 0:
        cond_mean_eta = np.zeros_like(cond_mean_eta)
    else:
        mask = np.ones(nb_parameters, dtype=bool)
        mask[ind_eta] = False
        cond_mean_eta[:, mask] = 0.0

    return {
        "mean_phi": mean_phi,
        "varList": varList,
        "suffStat": suffStat,
        "parpop": parpop,
        "allpar": allpar,
        "phiM": phiM,
        "cond_mean_phi": cond_mean_phi,
        "cond_var_phi": cond_var_phi,
        "cond_mean_psi": cond_mean_psi,
        "cond_mean_eta": cond_mean_eta,
    }
