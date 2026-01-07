import warnings
import numpy as np
from saemix.utils import cutoff, transphi


def compute_omega_safe(
    phi_samples: np.ndarray, mu: np.ndarray, min_eigenvalue: float = 1e-8
) -> np.ndarray:
    """
    Safely compute covariance matrix, ensuring positive definiteness.

    This function computes the sample covariance matrix and corrects it
    if necessary to ensure it is positive definite (all eigenvalues >= min_eigenvalue).

    Parameters
    ----------
    phi_samples : np.ndarray
        Individual parameter samples, shape (n_subjects, n_params)
    mu : np.ndarray
        Mean vector, shape (n_params,) or (n_subjects, n_params)
    min_eigenvalue : float
        Minimum eigenvalue threshold. Eigenvalues below this will be corrected.
        Default is 1e-8.

    Returns
    -------
    np.ndarray
        Positive definite covariance matrix, shape (n_params, n_params)

    Raises
    ------
    ValueError
        If the matrix cannot be corrected to be positive definite,
        or if inputs are invalid.

    Notes
    -----
    The function performs the following steps:
    1. Compute sample covariance matrix
    2. Check eigenvalues using symmetric eigendecomposition
    3. If any eigenvalue < min_eigenvalue, correct by setting to min_eigenvalue
    4. Reconstruct the matrix and ensure symmetry
    5. Verify Cholesky decomposition is possible

    **Validates: Requirements 5.3, 5.4**
    """
    # Input validation
    phi_samples = np.asarray(phi_samples)
    mu = np.asarray(mu)

    if phi_samples.ndim != 2:
        raise ValueError(
            f"[compute_omega_safe] phi_samples must be 2D array, got shape {phi_samples.shape}. "
            "Suggestion: Reshape to (n_subjects, n_params)."
        )

    n_subjects, n_params = phi_samples.shape

    if n_subjects == 0:
        raise ValueError(
            "[compute_omega_safe] phi_samples has 0 subjects. "
            "Context: Cannot compute covariance from empty data. "
            "Suggestion: Check data filtering and ensure subjects remain after processing."
        )

    # Handle mu shape - can be (n_params,) or (n_subjects, n_params)
    if mu.ndim == 1:
        if mu.shape[0] != n_params:
            raise ValueError(
                f"[compute_omega_safe] mu shape {mu.shape} incompatible with phi_samples shape {phi_samples.shape}. "
                "Suggestion: Ensure mu has n_params elements."
            )
        mu_expanded = mu
    elif mu.ndim == 2:
        if mu.shape[1] != n_params:
            raise ValueError(
                f"[compute_omega_safe] mu shape {mu.shape} incompatible with phi_samples shape {phi_samples.shape}. "
                "Suggestion: Ensure mu has n_params columns."
            )
        # Use mean of mu if it's per-subject
        mu_expanded = mu.mean(axis=0) if mu.shape[0] > 1 else mu[0]
    else:
        raise ValueError(
            f"[compute_omega_safe] mu must be 1D or 2D array, got {mu.ndim}D. "
            "Suggestion: Provide mu as shape (n_params,) or (n_subjects, n_params)."
        )

    # Compute centered data
    centered = phi_samples - mu_expanded

    # Check for NaN/Inf in inputs
    if not np.all(np.isfinite(centered)):
        n_nan = np.sum(np.isnan(centered))
        n_inf = np.sum(np.isinf(centered))
        raise ValueError(
            f"[compute_omega_safe] Input contains non-finite values. "
            f"Context: {n_nan} NaN values, {n_inf} Inf values. "
            f"phi_samples range: [{np.nanmin(phi_samples):.4g}, {np.nanmax(phi_samples):.4g}], "
            f"mu range: [{np.nanmin(mu):.4g}, {np.nanmax(mu):.4g}]. "
            "Suggestion: Check for numerical issues in parameter estimation."
        )

    # Compute sample covariance
    omega = np.dot(centered.T, centered) / max(n_subjects, 1)

    # Ensure symmetry (handle numerical precision issues)
    omega = (omega + omega.T) / 2

    # Check and correct positive definiteness
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(omega)

        if np.any(eigenvalues < min_eigenvalue):
            n_corrected = np.sum(eigenvalues < min_eigenvalue)
            min_eigenval = eigenvalues.min()
            warnings.warn(
                f"[compute_omega_safe] Covariance matrix has {n_corrected} small/negative eigenvalues "
                f"(min={min_eigenval:.2e}). Correcting to ensure positive definiteness. "
                "This may indicate near-collinearity in random effects or insufficient data.",
                UserWarning,
            )
            # Correct small/negative eigenvalues
            eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
            # Reconstruct matrix
            omega = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            # Ensure symmetry after reconstruction
            omega = (omega + omega.T) / 2

        # Verify Cholesky decomposition is possible
        np.linalg.cholesky(omega)

    except np.linalg.LinAlgError as e:
        raise ValueError(
            f"[compute_omega_safe] Failed to compute valid covariance matrix. "
            f"Original error: {e}. "
            f"Context: n_subjects={n_subjects}, n_params={n_params}, "
            f"omega diagonal range: [{np.diag(omega).min():.4g}, {np.diag(omega).max():.4g}]. "
            "Suggestion: Check model specification, ensure sufficient data, "
            "and verify covariance model structure."
        ) from e

    return omega


def _normalize_ytype(ytype, ntypes):
    if ytype is None:
        return None
    ytype_arr = np.asarray(ytype).astype(int)
    if ytype_arr.size == 0:
        return ytype_arr
    if ntypes > 1 and ytype_arr.min() >= 1 and ytype_arr.max() == ntypes:
        return ytype_arr - 1
    return ytype_arr


def _estimate_mean_phi(phiM, Uargs):
    nchains = Uargs["nchains"]
    N = Uargs["COV"].shape[0] if Uargs["COV"].size > 0 else int(phiM.shape[0] / nchains)
    nb_parameters = Uargs["nb_parameters"]
    if nchains <= 1:
        phi_mean = phiM.reshape((N, nb_parameters))
    else:
        phi_chain = phiM.reshape((nchains, N, nb_parameters))
        phi_mean = phi_chain.mean(axis=0)
    Mcovariates = Uargs["Mcovariates"]
    betaest_model = Uargs["betaest_model"]
    fixed_estim = Uargs["fixed_estim"]
    mean_phi = np.zeros_like(phi_mean)
    for j in range(nb_parameters):
        jcov = np.where(betaest_model[:, j] == 1)[0]
        if len(jcov) == 0:
            mean_phi[:, j] = phi_mean[:, j]
            continue
        X = Mcovariates[:, jcov]
        y = phi_mean[:, j]
        # Keep intercept fixed if requested
        if 0 in jcov and fixed_estim[j] == 0:
            intercept_idx = np.where(jcov == 0)[0][0]
            intercept = np.mean(y)
            X_adj = np.delete(X, intercept_idx, axis=1)
            if X_adj.size > 0:
                coef, _, _, _ = np.linalg.lstsq(X_adj, y - intercept, rcond=None)
                lambdaj = np.insert(coef, intercept_idx, intercept)
            else:
                lambdaj = np.array([intercept])
        else:
            lambdaj, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        mean_phi[:, j] = X @ lambdaj
    return mean_phi


def _estimate_residual(y, f, error_model, ytype, current_pres):
    eps = 1e-10
    error_model = (
        list(error_model) if isinstance(error_model, (list, tuple)) else [error_model]
    )
    ytype_norm = _normalize_ytype(ytype, len(error_model))
    pres = np.array(current_pres, dtype=float).copy()
    if len(error_model) == 1:
        em = error_model[0]
        if em == "constant" or em == "exponential":
            sig2 = np.mean((y - f) ** 2)
            pres[0] = np.sqrt(max(sig2, eps))
        elif em == "proportional":
            denom = np.maximum(np.abs(f), eps)
            sig2 = np.mean(((y - f) / denom) ** 2)
            pres[1] = np.sqrt(max(sig2, eps))
        elif em == "combined":
            from scipy.optimize import minimize

            def nll(ab):
                a = max(ab[0], eps)
                b = max(ab[1], eps)
                g2 = a**2 + (b * f) ** 2
                return np.sum(0.5 * ((y - f) ** 2) / g2 + 0.5 * np.log(g2))

            x0 = np.maximum(pres[:2], eps)
            res = minimize(
                nll, x0, method="L-BFGS-B", bounds=[(eps, None), (eps, None)]
            )
            pres[:2] = res.x
        return pres
    for ityp, em in enumerate(error_model):
        if ytype_norm is None:
            mask = slice(None)
        else:
            mask = ytype_norm == ityp
        if not np.any(mask):
            continue
        yi = y[mask]
        fi = f[mask]
        base = 2 * ityp
        if em == "constant" or em == "exponential":
            sig2 = np.mean((yi - fi) ** 2)
            pres[base] = np.sqrt(max(sig2, eps))
        elif em == "proportional":
            denom = np.maximum(np.abs(fi), eps)
            sig2 = np.mean(((yi - fi) / denom) ** 2)
            pres[base + 1] = np.sqrt(max(sig2, eps))
        elif em == "combined":
            from scipy.optimize import minimize

            def nll(ab):
                a = max(ab[0], eps)
                b = max(ab[1], eps)
                g2 = a**2 + (b * fi) ** 2
                return np.sum(0.5 * ((yi - fi) ** 2) / g2 + 0.5 * np.log(g2))

            x0 = np.maximum(pres[base : base + 2], eps)
            res = minimize(
                nll, x0, method="L-BFGS-B", bounds=[(eps, None), (eps, None)]
            )
            pres[base : base + 2] = res.x
    return pres


def mstep(
    kiter, Uargs, Dargs, opt, structural_model, phiM, varList, suffStat, mean_phi_init
):
    stepsize = opt["stepsize"][kiter - 1]

    mean_phi = mean_phi_init.copy()

    if stepsize > 0:
        nchains = Uargs["nchains"]
        N = Dargs["N"]
        nb_parameters = Uargs["nb_parameters"]
        if nchains <= 1:
            phi_chain = phiM.reshape((N, nb_parameters))[None, :, :]
        else:
            phi_chain = phiM.reshape((nchains, N, nb_parameters))
        phi_eta = (
            phi_chain[:, :, varList["ind_eta"]]
            if len(varList["ind_eta"]) > 0
            else np.zeros((nchains, N, 0))
        )
        stat1 = phi_eta.sum(axis=0)
        stat2 = np.zeros((len(varList["ind_eta"]), len(varList["ind_eta"])))
        stat3 = (phi_chain**2).sum(axis=0)
        for k in range(nchains):
            if len(varList["ind_eta"]) > 0:
                phik = phi_eta[k, :, :]
                stat2 += phik.T @ phik
        suffStat["statphi1"] = suffStat["statphi1"] + stepsize * (
            stat1 / nchains - suffStat["statphi1"]
        )
        suffStat["statphi2"] = suffStat["statphi2"] + stepsize * (
            stat2 / nchains - suffStat["statphi2"]
        )
        suffStat["statphi3"] = suffStat["statphi3"] + stepsize * (
            stat3 / nchains - suffStat["statphi3"]
        )

        mean_phi = _estimate_mean_phi(phiM, Uargs)

        # Update omega using accumulated sufficient statistics (statphi2)
        # This is the correct SAEM M-step: Omega = S_2^(k) (sufficient statistic)
        # NOT the current sample covariance
        omega_full = np.zeros_like(varList["omega"])
        ind_eta = varList["ind_eta"]
        if len(ind_eta) > 0:
            omega_eta = suffStat["statphi2"]
            omega_full[np.ix_(ind_eta, ind_eta)] = omega_eta
        omega_full = np.where(omega_full < 0, 1e-6, omega_full)
        omega_new = np.zeros_like(varList["omega"])
        indest_omega = Uargs.get("indest_omega", None)
        if indest_omega is not None:
            omega_new[indest_omega] = omega_full[indest_omega]
        else:
            omega_new = omega_full
        varList["omega"] = omega_new

        # Residual error update
        if Dargs.get("modeltype", "structural") == "structural":
            psiM = transphi(phiM, Dargs["transform_par"])
            fpred = structural_model(psiM, Dargs["IdM"], Dargs["XM"])
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
            new_pres = _estimate_residual(
                Dargs["yM"], fpred, error_model, ytype, varList["pres"]
            )
            if kiter <= opt["nbiter_sa"]:
                varList["pres"] = np.maximum(
                    varList["pres"] * opt["alpha1_sa"], new_pres
                )
            else:
                varList["pres"] = varList["pres"] + stepsize * (
                    new_pres - varList["pres"]
                )

    return {
        "varList": varList,
        "suffStat": suffStat,
        "mean_phi": mean_phi,
    }