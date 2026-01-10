import warnings
from typing import Optional

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import t as student_t

from saemix.algorithm.map_estimation import error_function
from saemix.utils import cutoff, transphi


def _log_mean_exp(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Compute log(mean(exp(x))) in a numerically stable way using log-sum-exp.

    Parameters
    ----------
    x : np.ndarray
        Input array
    axis : int, optional
        Axis along which to compute mean

    Returns
    -------
    np.ndarray
        log(mean(exp(x))) along specified axis
    """
    x = np.asarray(x)
    if axis is None:
        x = x.ravel()
        axis = 0
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted, axis=axis, keepdims=True)
    log_sum_exp = np.log(sum_exp) + x_max
    # Compute mean: log(mean) = log(sum_exp / N) = log_sum_exp - log(N)
    n = x.shape[axis]
    result = log_sum_exp - np.log(n)
    # Remove the dimension that was kept
    if axis is not None:
        result = result.squeeze(axis=axis)
    return result


def compute_log_likelihood_safe(
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    sigma: float,
    log_eps: float = -700,
    iteration: int = None,
    param_values: dict = None,
) -> float:
    """
    Safely compute log-likelihood with NaN/Inf checking and diagnostics.

    This function computes the Gaussian log-likelihood for observed vs predicted
    values, with comprehensive error checking and diagnostic information.

    Parameters
    ----------
    y_obs : np.ndarray
        Observed values, shape (n_obs,)
    y_pred : np.ndarray
        Predicted values, shape (n_obs,)
    sigma : float
        Residual standard deviation (must be positive)
    log_eps : float
        Lower bound for log-likelihood to prevent numerical underflow.
        Default is -700 (approximately log of smallest positive float64).
    iteration : int, optional
        Current iteration number for diagnostic messages.
    param_values : dict, optional
        Current parameter values for diagnostic messages.

    Returns
    -------
    float
        Log-likelihood value (finite, >= log_eps)

    Raises
    ------
    ValueError
        If sigma <= 0, or if computation produces NaN or +Inf.

    Notes
    -----
    The log-likelihood is computed as:
        LL = -0.5 * n * log(2 * pi * sigma^2) - 0.5 * sum((y_obs - y_pred)^2) / sigma^2

    **Validates: Requirements 5.2, 5.5**
    """
    # Convert inputs to arrays
    y_obs = np.asarray(y_obs).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Input validation
    if y_obs.shape != y_pred.shape:
        raise ValueError(
            f"[compute_log_likelihood_safe] Shape mismatch: y_obs {y_obs.shape} vs y_pred {y_pred.shape}. "
            "Suggestion: Ensure observed and predicted arrays have the same length."
        )

    if sigma <= 0:
        iter_info = f" at iteration {iteration}" if iteration is not None else ""
        param_info = f", parameters: {param_values}" if param_values else ""
        raise ValueError(
            f"[compute_log_likelihood_safe] Sigma must be positive, got {sigma}{iter_info}{param_info}. "
            "Suggestion: Check residual error estimation and ensure error model is appropriate."
        )

    n = len(y_obs)
    if n == 0:
        raise ValueError(
            "[compute_log_likelihood_safe] Empty observation array. "
            "Suggestion: Check data filtering and ensure observations remain."
        )

    # Check for non-finite values in inputs
    if not np.all(np.isfinite(y_obs)):
        n_nan = np.sum(np.isnan(y_obs))
        n_inf = np.sum(np.isinf(y_obs))
        raise ValueError(
            f"[compute_log_likelihood_safe] y_obs contains non-finite values. "
            f"Context: {n_nan} NaN, {n_inf} Inf values. "
            "Suggestion: Check data preprocessing and missing value handling."
        )

    if not np.all(np.isfinite(y_pred)):
        n_nan = np.sum(np.isnan(y_pred))
        n_inf = np.sum(np.isinf(y_pred))
        iter_info = f" at iteration {iteration}" if iteration is not None else ""
        raise ValueError(
            f"[compute_log_likelihood_safe] y_pred contains non-finite values{iter_info}. "
            f"Context: {n_nan} NaN, {n_inf} Inf values, "
            f"y_pred range: [{np.nanmin(y_pred):.4g}, {np.nanmax(y_pred):.4g}]. "
            "Suggestion: Check model function for numerical stability, "
            "especially with extreme parameter values."
        )

    # Compute residuals
    residuals = y_obs - y_pred

    # Compute log-likelihood
    # LL = -0.5 * n * log(2 * pi * sigma^2) - 0.5 * sum(residuals^2) / sigma^2
    sigma_sq = sigma**2
    ll = -0.5 * n * np.log(2 * np.pi * sigma_sq) - 0.5 * np.sum(residuals**2) / sigma_sq

    # Check for NaN result
    if np.isnan(ll):
        iter_info = f" at iteration {iteration}" if iteration is not None else ""
        param_info = f", parameters: {param_values}" if param_values else ""
        raise ValueError(
            f"[compute_log_likelihood_safe] Log-likelihood is NaN{iter_info}. "
            f"Context: y_obs range: [{y_obs.min():.4g}, {y_obs.max():.4g}], "
            f"y_pred range: [{y_pred.min():.4g}, {y_pred.max():.4g}], "
            f"sigma: {sigma:.4g}, residuals range: [{residuals.min():.4g}, {residuals.max():.4g}]{param_info}. "
            "Suggestion: Check for numerical overflow in residual computation."
        )

    # Check for +Inf (invalid - would indicate impossible perfect fit)
    if np.isinf(ll) and ll > 0:
        iter_info = f" at iteration {iteration}" if iteration is not None else ""
        raise ValueError(
            f"[compute_log_likelihood_safe] Log-likelihood is +Inf, which is invalid{iter_info}. "
            f"Context: sigma={sigma:.4g}, residuals sum of squares={np.sum(residuals**2):.4g}. "
            "Suggestion: Check for zero residuals or invalid sigma value."
        )

    # Apply lower bound to prevent numerical issues (very negative LL is valid but may cause issues)
    if ll < log_eps:
        warnings.warn(
            f"[compute_log_likelihood_safe] Log-likelihood {ll:.4g} truncated to {log_eps} "
            "to prevent numerical underflow.",
            UserWarning,
        )
        ll = log_eps

    return ll


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
    index = data.data["index"].values
    ytype = data.data["ytype"].values if "ytype" in data.data.columns else None
    ytype_norm = _normalize_ytype(ytype, len(model.error_model))

    if nphi1 == 0:
        psi = transphi(cond_mean_phi, model.transform_par)
        f = model.model(psi, index, xind)
        if model.modeltype == "structural":
            if len(model.error_model) > 1 and ytype_norm is not None:
                for ityp, em in enumerate(model.error_model):
                    if em == "exponential":
                        mask = ytype_norm == ityp
                        if np.any(mask):
                            f[mask] = np.log(cutoff(f[mask]))
            elif len(model.error_model) == 1 and model.error_model[0] == "exponential":
                f = np.log(cutoff(f))
            g = error_function(f, res.respar, model.error_model, ytype_norm)
            ll = -0.5 * np.sum(
                ((yobs - f) / g) ** 2 + 2 * np.log(g) + np.log(2 * np.pi)
            )
        else:
            ll = np.sum(f)
        res.ll_is = ll
        npar = (
            res.npar_est
            if getattr(res, "npar_est", None) is not None
            else _compute_npar_est(saemix_object)
        )
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

    nmc_is = int(opts.get("nmc_is", 5000))
    MM = 100
    KM = max(1, int(np.round(nmc_is / MM)))

    cond_var_phi1 = cutoff(cond_var_phi[:, i1_omega2])
    mean_phi1 = mean_phi[:, i1_omega2]

    mean_phiM1 = np.repeat(mean_phi1, MM, axis=0)
    mtild_phiM1 = np.repeat(cond_mean_phi[:, i1_omega2], MM, axis=0)
    stild_phiM1 = np.repeat(np.sqrt(cond_var_phi1), MM, axis=0)
    phiM = np.repeat(cond_mean_phi, MM, axis=0)

    # IdM maps each observation to the corresponding row in phiM
    # phiM layout: rows 0 to N-1 are sample 0, rows N to 2N-1 are sample 1, etc.
    # Formula: IdM = sample_id * N + subject_id
    # This matches R: rep(0:(MM-1), each=ntot.obs) * N + rep(index, MM)
    sample_block = np.repeat(np.arange(MM), len(index))  # 0,0,...,0,1,1,...,1,...
    subject_ids = np.tile(index, MM)  # index repeated MM times
    IdM = sample_block * data.n_subjects + subject_ids
    yM = np.tile(yobs, MM)
    XM = np.tile(xind, (MM, 1))
    ytypeM = np.tile(ytype_norm, MM) if ytype_norm is not None else None

    log_const = _log_const_exponential(yobs, ytype_norm, model.error_model, yorig=yorig)
    c2 = np.log(np.linalg.det(omega_sub)) + nphi1 * np.log(2 * np.pi)
    c1 = np.log(2 * np.pi)

    meana = np.zeros(data.n_subjects)
    LL = np.zeros(KM)

    for km in range(1, KM + 1):
        r = student_t.rvs(df=opts.get("nu_is", 4), size=(data.n_subjects * MM, nphi1))
        phiM1 = mtild_phiM1 + stild_phiM1 * r
        dphiM = phiM1 - mean_phiM1
        d2 = -0.5 * (np.sum(dphiM * (dphiM @ IOmega), axis=1) + c2)
        e2 = d2.reshape(data.n_subjects, MM)

        log_tpdf = student_t.logpdf(r, df=opts.get("nu_is", 4))
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

        sume = e1 + e2 - e3

        # Use log-sum-exp for numerical stability
        log_newa = _log_mean_exp(sume, axis=1)
        newa = np.exp(log_newa)
        meana = meana + (newa - meana) / km
        LL[km - 1] = np.sum(np.log(cutoff(meana))) + log_const

    ll_is = LL[-1]
    res.ll_is = ll_is
    npar = (
        res.npar_est
        if getattr(res, "npar_est", None) is not None
        else _compute_npar_est(saemix_object)
    )
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
    index = data.data["index"].values
    ytype = data.data["ytype"].values if "ytype" in data.data.columns else None
    ytype_norm = _normalize_ytype(ytype, len(model.error_model))

    if nphi1 == 0:
        return llis_saemix(saemix_object)

    nnodes = int(opts.get("nnodes_gq", 12))
    nsd_gq = float(opts.get("nsd_gq", 4))

    max_grid = nnodes**nphi1
    if max_grid > 200000:
        raise ValueError(
            "Gaussian quadrature grid too large; reduce nnodes_gq or random effect dimension."
        )

    nodes_1d, weights_1d = leggauss(nnodes)
    grids = np.meshgrid(*([nodes_1d] * nphi1), indexing="ij")
    x = np.stack([g.ravel() for g in grids], axis=1)
    wgrids = np.meshgrid(*([weights_1d] * nphi1), indexing="ij")
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
                        mask = ytype_norm == ityp
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
    ll_gq = (
        (-S / 2)
        + np.sum(np.log(cutoff(Q)) + np.sum(np.log(cutoff(b)), axis=1))
        + log_const
    )

    res.ll_gq = ll_gq
    npar = (
        res.npar_est
        if getattr(res, "npar_est", None) is not None
        else _compute_npar_est(saemix_object)
    )
    res.aic_gq = -2 * ll_gq + 2 * npar
    res.bic_gq = -2 * ll_gq + np.log(data.n_subjects) * npar
    res.ll = ll_gq
    res.aic = res.aic_gq
    res.bic = res.bic_gq
    return saemix_object
