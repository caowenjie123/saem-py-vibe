import numpy as np
from saemix.data import SaemixData
from saemix.model import SaemixModel
from saemix.utils import transpsi, mydiag


def initialise_main_algo(saemix_data: SaemixData, saemix_model: SaemixModel, saemix_options: dict):
    structural_model = saemix_model.model
    nb_parameters = saemix_model.n_parameters
    N = saemix_data.n_subjects
    
    if saemix_model.modeltype == "structural":
        if len(saemix_model.error_init) > 0:
            pres = np.array(saemix_model.error_init, dtype=float)
        else:
            pres = np.array([1.0, 0.0], dtype=float)
    else:
        pres = np.array([], dtype=float)
    
    i0_omega2 = np.where(1 - np.diag(saemix_model.covariance_model) > 0)[0]
    indest_omega = np.where(saemix_model.covariance_model > 0)
    i1_omega2 = saemix_model.indx_omega
    ind_res = saemix_model.indx_res
    
    id_col = saemix_data.data[saemix_data.name_group].values
    if len(saemix_data.name_covariates) == 0:
        tab = {'id': id_col}
    else:
        tab = {'id': id_col}
        for cov in saemix_data.name_covariates:
            tab[cov] = saemix_data.data[cov].values
    
    unique_indices = np.unique(saemix_data.data['index'].values)
    ncov = len(saemix_data.name_covariates)
    Mcovariates = np.ones((N, 1 + ncov))
    if ncov > 0:
        for idx, orig_idx in enumerate(unique_indices):
            cov_values = saemix_data.data[saemix_data.data['index'] == orig_idx][saemix_data.name_covariates].iloc[0].values
            Mcovariates[idx, 1:] = cov_values
    
    j_cov = np.where(np.sum(saemix_model.betaest_model, axis=1) > 0)[0]
    betaest_model = saemix_model.betaest_model[j_cov, :]
    Mcovariates = Mcovariates[:, j_cov]
    
    saemix_model.betaest_model = betaest_model
    if betaest_model.shape[0] > 1:
        temp1 = betaest_model[1:, :]
        saemix_model.covariate_model = temp1
    else:
        saemix_model.covariate_model = np.zeros((0, nb_parameters))
    
    fixedpsi_ini = saemix_model.psi0[0, :]
    betaI_ini = transpsi(fixedpsi_ini.reshape(1, -1), saemix_model.transform_par).flatten()
    fixed_ini = np.zeros_like(betaest_model)
    fixed_ini[0, :] = betaI_ini
    
    nr_psi0 = saemix_model.psi0.shape[0]
    nr_cov = betaest_model.shape[0]
    
    if nr_psi0 > nr_cov:
        saemix_model.psi0 = saemix_model.psi0[:nr_cov, :]
        nr_psi0 = saemix_model.psi0.shape[0]
    
    if nr_psi0 < nr_cov:
        psi1 = saemix_model.psi0[-1, :]
        for j in range(nr_psi0, nr_cov):
            saemix_model.psi0 = np.vstack([saemix_model.psi0, psi1.reshape(1, -1)])
        nr_psi0 = saemix_model.psi0.shape[0]
    
    if nr_psi0 > 1:
        fixed_ini[1:nr_psi0, :] = saemix_model.psi0[1:nr_psi0, :]
    
    covariate_estim = betaest_model.copy()
    covariate_estim[0, :] = saemix_model.fixed_estim
    
    betas_ini = fixed_ini[saemix_model.betaest_model > 0]
    betas_ini = betas_ini.reshape(-1, 1)
    
    nb_betas = int(np.sum(saemix_model.betaest_model))
    ind_covariate = np.where(saemix_model.betaest_model == 1)
    
    LCOV = np.zeros((nb_betas, nb_parameters))
    MCOV = np.zeros((nb_betas, nb_parameters))
    COV = np.zeros((N, 0))
    pfix = np.zeros(nb_parameters)
    mean_phi = np.zeros((N, nb_parameters))
    betas = []
    
    j1 = 0
    for j in range(nb_parameters):
        jcov = np.where(betaest_model[:, j] == 1)[0]
        lambdaj = fixed_ini[jcov, j]
        aj = Mcovariates[:, jcov]
        COV = np.hstack([COV, aj])
        nlj = len(lambdaj)
        j2 = j1 + nlj
        LCOV[j1:j2, j] = 1
        betas.extend(lambdaj.tolist())
        j1 = j2
        if len(jcov) == 1:
            mean_phi[:, j] = aj.flatten() * lambdaj[0]
        else:
            mean_phi[:, j] = (aj @ lambdaj).flatten()
        pfix[j] = len(lambdaj)
    
    betas = np.array(betas).reshape(-1, 1) if len(betas) > 0 else np.zeros((0, 1))
    if betas.size > 0:
        MCOV[LCOV == 1] = betas.flatten()
    
    if nb_parameters > 1:
        indx_betaI = np.cumsum(np.concatenate([[0], pfix[:-1]]))
    else:
        indx_betaI = np.array([0])
    
    indx_betaC = np.where(saemix_model.betaest_model[1:, :].flatten() > 0)[0] + nb_parameters
    
    indx_fix10 = np.where((saemix_model.fixed_estim > 0) & (np.diag(saemix_model.covariance_model) == 0))[0]
    indx_fix11 = np.where((saemix_model.fixed_estim > 0) & (np.diag(saemix_model.covariance_model) > 0))[0]
    indx_fix1 = np.concatenate([indx_fix10, indx_fix11])
    indx_fix0 = np.where(saemix_model.fixed_estim == 0)[0]
    
    nb_etas = len(i1_omega2)
    
    omega = saemix_model.omega_init.copy()
    omega_eta = omega[np.ix_(i1_omega2, i1_omega2)] if len(i1_omega2) > 0 else np.zeros((0, 0))
    diag_omega = np.diag(omega) if omega.size > 0 else np.array([])
    if len(i1_omega2) > 0:
        base = np.sqrt(np.maximum(np.diag(omega_eta), 1e-10)) * saemix_options.get('rw_init', 0.5)
        domega2 = np.tile(base.reshape(-1, 1), (1, len(i1_omega2)))
    else:
        domega2 = np.zeros((0, 0))
    
    varList = {
        'pres': pres,
        'ind0_eta': i0_omega2,
        'ind_eta': i1_omega2,
        'omega': omega,
        'MCOV': MCOV,
        'domega2': domega2,
        'diag_omega': diag_omega,
    }
    
    Uargs = {
        'nchains': saemix_options['nb_chains'],
        'nb_parameters': nb_parameters,
        'nb_betas': nb_betas,
        'nb_etas': nb_etas,
        'nb_parest': None,
        'indx_betaC': indx_betaC,
        'indx_betaI': indx_betaI,
        'ind_res': ind_res,
        'indest_omega': indest_omega,
        'i0_omega2': i0_omega2,
        'i1_omega2': i1_omega2,
        'j_covariate': j_cov,
        'ind_fix10': indx_fix10,
        'ind_fix11': indx_fix11,
        'ind_fix1': indx_fix1,
        'ind_fix0': indx_fix0,
        'COV': COV,
        'LCOV': LCOV,
        'Mcovariates': Mcovariates,
        'betaest_model': betaest_model,
        'fixed_estim': saemix_model.fixed_estim,
    }
    
    nbiter_saemix = saemix_options['nbiter_saemix']
    nbiter_sa = saemix_options['nbiter_sa']
    nbiter_tot = nbiter_saemix[0] + nbiter_saemix[1]
    
    stepsize = np.ones(nbiter_tot)
    stepsize[:nbiter_sa] = 1.0
    alpha1_sa = saemix_options['alpha_sa']
    for k in range(nbiter_sa, nbiter_tot):
        stepsize[k] = 1.0 / (k - nbiter_sa + 1)
    
    opt = {
        'stepsize_rw': saemix_options['stepsize_rw'],
        'stepsize': stepsize,
        'proba_mcmc': saemix_options['proba_mcmc'],
        'nbiter_mcmc': saemix_options['nbiter_mcmc'],
        'nbiter_sa': nbiter_sa,
        'nbiter_map': saemix_options['nbiter_map'],
        'alpha1_sa': alpha1_sa,
        'alpha0_sa': 10**(-3 / nbiter_sa),
        'nbiter_saemix': nbiter_saemix,
        'maxim_maxiter': saemix_options['maxim_maxiter'],
        'flag_fmin': True,
    }
    
    index = saemix_data.data['index'].values
    nobs = len(index)
    nb_chains = saemix_options['nb_chains']
    IdM = np.tile(index, nb_chains) + np.repeat(np.arange(nb_chains) * N, nobs)
    XM = saemix_data.data[saemix_data.name_predictors].values
    XM = np.tile(XM, (nb_chains, 1))
    yM = np.tile(saemix_data.data[saemix_data.name_response].values, nb_chains)
    ytype = None
    if 'ytype' in saemix_data.data.columns:
        ytype = np.tile(saemix_data.data['ytype'].values, nb_chains)
    NM = N * nb_chains
    
    etype_exp = [i for i, em in enumerate(saemix_model.error_model) if em == "exponential"]
    Dargs = {
        'IdM': IdM,
        'XM': XM,
        'yM': yM,
        'NM': NM,
        'N': N,
        'nobs': saemix_data.n_total_obs,
        'yobs': saemix_data.data[saemix_data.name_response].values,
        'transform_par': saemix_model.transform_par,
        'error_model': saemix_model.error_model,
        'structural_model': structural_model,
        'modeltype': saemix_model.modeltype,
        'ytype': ytype,
        'etype_exp': etype_exp,
    }

    # Number of estimated parameters (for AIC/BIC)
    covariate_estim = betaest_model.copy()
    covariate_estim[0, :] = saemix_model.fixed_estim
    omega_upper = np.triu(saemix_model.covariance_model, k=0)
    n_omega = int(np.sum(omega_upper != 0))
    if saemix_model.modeltype == "structural":
        n_res = 0
        for em in saemix_model.error_model:
            n_res += 2 if em == "combined" else 1
    else:
        n_res = 0
    nb_parest = int(np.sum(covariate_estim)) + n_omega + n_res
    Uargs['nb_parest'] = nb_parest
    
    # Initialisation of phiM
    mean_phiM = np.tile(mean_phi, (nb_chains, 1))
    phiM = mean_phiM.copy()
    if len(i1_omega2) > 0:
        try:
            chol_omega = np.linalg.cholesky(omega_eta)
        except np.linalg.LinAlgError:
            chol_omega = np.eye(len(i1_omega2))
        etaM = 0.5 * np.random.randn(NM, len(i1_omega2)) @ chol_omega.T
        phiM[:, i1_omega2] = mean_phiM[:, i1_omega2] + etaM
    
    # Ensure initial parameters give finite predictions
    if saemix_model.modeltype == "structural":
        max_tries = 100
        tries = 0
        while tries < max_tries:
            from saemix.utils import transphi
            psiM = transphi(phiM, saemix_model.transform_par)
            fpred = structural_model(psiM, IdM, XM)
            invalid = ~np.isfinite(fpred)
            if not np.any(invalid):
                break
            bad_ids = np.unique(IdM[invalid])
            if len(bad_ids) == 0:
                break
            if len(i1_omega2) > 0:
                etaM = 0.5 * np.random.randn(len(bad_ids), len(i1_omega2)) @ chol_omega.T
                phiM[np.ix_(bad_ids, i1_omega2)] = mean_phiM[np.ix_(bad_ids, i1_omega2)] + etaM
            tries += 1
        if tries >= max_tries:
            raise RuntimeError("Failed to find a valid initial parameter guess for structural model.")
    
    return {
        'saemix_model': saemix_model,
        'Dargs': Dargs,
        'Uargs': Uargs,
        'varList': varList,
        'opt': opt,
        'mean_phi': mean_phi,
        'betas_ini': betas_ini,
        'betas': betas,
        'phiM': phiM,
    }