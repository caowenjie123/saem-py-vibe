import numpy as np
from scipy.stats import t as student_t
from numpy.polynomial.legendre import leggauss
from saemix.utils import transphi, cutoff
from saemix.algorithm.map_estimation import error_function


def _normalize_ytype(ytype, ntypes):
    if ytype is None:
        return None
    ytype_arr = np.asarray(ytype).astype(int)
    if ytype_arr.size == 0:
        return ytype_arr
    if ntypes > 1 and ytype_arr.min() >= 1 and ytype_arr.max() == ntypes:
        return ytype_arr - 1
    return ytype_arr


def _log_const_exponential(yobs, ytype, error_model, yorig=None):
    idx_exp = [i for i, em in enumerate(error_model) if em == "exponential"]
    if not idx_exp or ytype is None:
        return 0.0
    ytype_norm = _normalize_ytype(ytype, len(error_model))
    ybase = yorig if yorig is not None else yobs
    mask = np.isin(ytype_norm, idx_exp)
    if np.any(mask):
        return -np.sum(ybase[mask])
    return 0.0


def _compute_npar_est(saemix_object):
    model = saemix_object.model
    betaest_model = getattr(model, "betaest_model", None)
    if betaest_model is None:
        betaest_model = np.ones((1, model.n_parameters))
    covariate_estim = betaest_model.copy()
    covariate_estim[0, :] = model.fixed_estim
    omega_upper = np.triu(model.covariance_model, k=0)
    n_omega = int(np.sum(omega_upper != 0))
    n_res = 0
    if model.modeltype == "structural":
        for em in model.error_model:
            n_res += 2 if em == "combined" else 1
    return int(np.sum(covariate_estim)) + n_omega + n_res


def llis_saemix(saemix_object):
    model = saemix_object.model
    data = saemix_object.data
    res = saemix_object.results
    opts = saemix_object.options
    
    if res.cond_mean_phi is None or res.cond_var_phi is None:
        raise ValueError("cond_mean_phi/cond_var_phi not available. Run SAEM first.")
    
    cond_mean_phi = np.asarray(res.cond_mean_phi)
    cond_var_phi = np.asarray(res.cond_var_phi)
    mean_phi = np.asarray(res.mean_phi)
    
    i1_omega2 = model.indx_omega
    nphi1 = len(i1_omega2)
    yobs = data.data[data.name_response].values
    yorig = getattr(data, "yorig", None)
    xind = data.data[data.name_predictors].values
    index = data.data['index'].values
    ytype = data.data['ytype'].values if 'ytype' in data.data.columns else None
    ytype_norm = _normalize_ytype(ytype, len(model.error_model))
    
    if nphi1 == 0:
        psi = transphi(cond_mean_phi, model.transform_par)
        f = model.model(psi, index, xind)
        if model.modeltype == "structural":
            if len(model.error_model) > 1 and ytype_norm is not None:
                for ityp, em in enumerate(model.error_model):
                    if em == "exponential":
                        mask = (ytype_norm == ityp)
                        if np.any(mask):
                            f[mask] = np.log(cutoff(f[mask]))
            elif len(model.error_model) == 1 and model.error_model[0] == "exponential":
                f = np.log(cutoff(f))
            g = error_function(f, res.respar, model.error_model, ytype_norm)
            ll = -0.5 * np.sum(((yobs - f) / g) ** 2 + 2 * np.log(g) + np.log(2 * np.pi))
        else:
            ll = np.sum(f)
        res.ll_is = ll
        npar = res.npar_est if getattr(res, "npar_est", None) is not None else _compute_npar_est(saemix_object)
        res.aic_is = -2 * ll + 2 * npar
        res.bic_is = -2 * ll + np.log(data.n_subjects) * npar
        res.ll = ll
        res.aic = res.aic_is
        res.bic = res.bic_is
        return saemix_object
    
    Omega = res.omega
    pres = res.respar
    omega_sub = Omega[np.ix_(i1_omega2, i1_omega2)]
    try:
        IOmega = np.linalg.inv(omega_sub)
    except np.linalg.LinAlgError:
        IOmega = np.eye(nphi1)
    
    nmc_is = int(opts.get('nmc_is', 5000))
    MM = 100
    KM = max(1, int(np.round(nmc_is / MM)))
    
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
    
    log_const = _log_const_exponential(yobs, ytype_norm, model.error_model, yorig=yorig)
    c2 = np.log(np.linalg.det(omega_sub)) + nphi1 * np.log(2 * np.pi)
    c1 = np.log(2 * np.pi)
    
    meana = np.zeros(data.n_subjects)
    LL = np.zeros(KM)
    
    for km in range(1, KM + 1):
        r = student_t.rvs(df=opts.get('nu_is', 4), size=(data.n_subjects * MM, nphi1))
        phiM1 = mtild_phiM1 + stild_phiM1 * r
        dphiM = phiM1 - mean_phiM1
        d2 = -0.5 * (np.sum(dphiM * (dphiM @ IOmega), axis=1) + c2)
        e2 = d2.reshape(data.n_subjects, MM)
        
        log_tpdf = student_t.logpdf(r, df=opts.get('nu_is', 4))
        pitild_phi1 = np.sum(log_tpdf, axis=1)
        e3 = pitild_phi1.reshape(data.n_subjects, MM) - np.repeat(
            0.5 * np.sum(np.log(cond_var_phi1), axis=1)[:, None], MM, axis=1
        )
        
        phiM[:, i1_omega2] = phiM1
        psiM = transphi(phiM, model.transform_par)
        f = model.model(psiM, IdM, XM)
        
        if model.modeltype == "structural":
            if len(model.error_model) > 1 and ytypeM is not None:
                for ityp, em in enumerate(model.error_model):
                    if em == "exponential":
                        mask = (ytypeM == ityp)
                        if np.any(mask):
                            f[mask] = np.log(cutoff(f[mask]))
            elif len(model.error_model) == 1 and model.error_model[0] == "exponential":
                f = np.log(cutoff(f))
            g = error_function(f, pres, model.error_model, ytypeM)
            dyf = -0.5 * ((yM - f) / g) ** 2 - np.log(g) - 0.5 * c1
        else:
            dyf = f
        
        e1 = np.bincount(IdM, weights=dyf, minlength=data.n_subjects * MM).reshape(data.n_subjects, MM)
        sume = e1 + e2 - e3
        newa = np.mean(np.exp(sume), axis=1)
        meana = meana + (newa - meana) / km
        LL[km - 1] = np.sum(np.log(cutoff(meana))) + log_const
    
    ll_is = LL[-1]
    res.ll_is = ll_is
    npar = res.npar_est if getattr(res, "npar_est", None) is not None else _compute_npar_est(saemix_object)
    res.aic_is = -2 * ll_is + 2 * npar
    res.bic_is = -2 * ll_is + np.log(data.n_subjects) * npar
    res.ll = ll_is
    res.aic = res.aic_is
    res.bic = res.bic_is
    return saemix_object


def llgq_saemix(saemix_object):
    model = saemix_object.model
    data = saemix_object.data
    res = saemix_object.results
    opts = saemix_object.options
    
    if res.cond_mean_phi is None or res.cond_var_phi is None:
        raise ValueError("cond_mean_phi/cond_var_phi not available. Run SAEM first.")
    
    cond_mean_phi = np.asarray(res.cond_mean_phi)
    cond_var_phi = np.asarray(res.cond_var_phi)
    mean_phi = np.asarray(res.mean_phi)
    
    i1_omega2 = model.indx_omega
    nphi1 = len(i1_omega2)
    yobs = data.data[data.name_response].values
    yorig = getattr(data, "yorig", None)
    xind = data.data[data.name_predictors].values
    index = data.data['index'].values
    ytype = data.data['ytype'].values if 'ytype' in data.data.columns else None
    ytype_norm = _normalize_ytype(ytype, len(model.error_model))
    
    if nphi1 == 0:
        return llis_saemix(saemix_object)
    
    nnodes = int(opts.get('nnodes_gq', 12))
    nsd_gq = float(opts.get('nsd_gq', 4))
    
    max_grid = nnodes ** nphi1
    if max_grid > 200000:
        raise ValueError("Gaussian quadrature grid too large; reduce nnodes_gq or random effect dimension.")
    
    nodes_1d, weights_1d = leggauss(nnodes)
    grids = np.meshgrid(*([nodes_1d] * nphi1), indexing='ij')
    x = np.stack([g.ravel() for g in grids], axis=1)
    wgrids = np.meshgrid(*([weights_1d] * nphi1), indexing='ij')
    w = np.prod(np.stack(wgrids, axis=-1), axis=-1).ravel()
    
    Omega = res.omega
    pres = res.respar
    omega_sub = Omega[np.ix_(i1_omega2, i1_omega2)]
    try:
        IOmega = np.linalg.inv(omega_sub)
    except np.linalg.LinAlgError:
        IOmega = np.eye(nphi1)
    
    condsd = np.sqrt(cutoff(cond_var_phi[:, i1_omega2]))
    xmin = cond_mean_phi[:, i1_omega2] - nsd_gq * condsd
    xmax = cond_mean_phi[:, i1_omega2] + nsd_gq * condsd
    a = (xmin + xmax) / 2
    b = (xmax - xmin) / 2
    
    log_const = _log_const_exponential(yobs, ytype_norm, model.error_model, yorig=yorig)
    
    Q = np.zeros(data.n_subjects)
    for j in range(x.shape[0]):
        phi = mean_phi.copy()
        phi[:, i1_omega2] = a + b * x[j, :]
        psi = transphi(phi, model.transform_par)
        f = model.model(psi, index, xind)
        if model.modeltype == "structural":
            if len(model.error_model) > 1 and ytype_norm is not None:
                for ityp, em in enumerate(model.error_model):
                    if em == "exponential":
                        mask = (ytype_norm == ityp)
                        if np.any(mask):
                            f[mask] = np.log(cutoff(f[mask]))
            elif len(model.error_model) == 1 and model.error_model[0] == "exponential":
                f = np.log(cutoff(f))
            g = error_function(f, pres, model.error_model, ytype_norm)
            dyf = -0.5 * ((yobs - f) / g) ** 2 - np.log(g)
        else:
            dyf = f
        ly = np.bincount(index, weights=dyf, minlength=data.n_subjects)
        dphi1 = phi[:, i1_omega2] - mean_phi[:, i1_omega2]
        lphi1 = -0.5 * np.sum((dphi1 @ IOmega) * dphi1, axis=1)
        ltot = ly + lphi1
        ltot[~np.isfinite(ltot)] = -np.inf
        Q += w[j] * np.exp(ltot)
    
    S = (
        data.n_subjects * np.log(np.linalg.det(omega_sub))
        + data.n_subjects * nphi1 * np.log(2 * np.pi)
        + data.n_total_obs * np.log(2 * np.pi)
    )
    ll_gq = (-S / 2) + np.sum(np.log(cutoff(Q)) + np.sum(np.log(cutoff(b)), axis=1)) + log_const
    
    res.ll_gq = ll_gq
    npar = res.npar_est if getattr(res, "npar_est", None) is not None else _compute_npar_est(saemix_object)
    res.aic_gq = -2 * ll_gq + 2 * npar
    res.bic_gq = -2 * ll_gq + np.log(data.n_subjects) * npar
    res.ll = ll_gq
    res.aic = res.aic_gq
    res.bic = res.bic_gq
    return saemix_object
