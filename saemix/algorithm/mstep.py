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
    """
    M-step of the SAEM algorithm.

    This implements the stochastic approximation update of sufficient statistics
    and the maximization step for population parameters (fixed effects, omega, residual error).

    The omega update follows the R saemix implementation:
    omega = statphi2/N + e1_phi'*e1_phi/N - statphi1'*e1_phi/N - e1_phi'*statphi1/N

    where e1_phi is the covariate effect on random effect parameters.

    Parameters
    ----------
    kiter : int
        Current iteration number (1-based)
    Uargs : dict
        Algorithm arguments (indices, covariates, etc.)
    Dargs : dict
        Data arguments
    opt : dict
        Algorithm options
    structural_model : callable
        Structural model function
    phiM : np.ndarray
        Current individual parameters (all chains)
    varList : dict
        Variability parameters (omega, pres, etc.)
    suffStat : dict
        Sufficient statistics
    mean_phi_init : np.ndarray
        Initial mean phi values

    Returns
    -------
    dict
        Updated varList, suffStat, and mean_phi
    """
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

        ind_eta = varList["ind_eta"]
        nb_etas = len(ind_eta)

        phi_eta = phi_chain[:, :, ind_eta] if nb_etas > 0 else np.zeros((nchains, N, 0))

        # Compute current iteration statistics
        stat1 = phi_eta.sum(axis=0)  # shape: (N, nb_etas)
        stat2 = np.zeros((nb_etas, nb_etas))
        stat3 = (phi_chain**2).sum(axis=0)  # shape: (N, nb_parameters)

        for k in range(nchains):
            if nb_etas > 0:
                phik = phi_eta[k, :, :]  # shape: (N, nb_etas)
                stat2 += phik.T @ phik

        # Update sufficient statistics using stochastic approximation
        suffStat["statphi1"] = suffStat["statphi1"] + stepsize * (
            stat1 / nchains - suffStat["statphi1"]
        )
        suffStat["statphi2"] = suffStat["statphi2"] + stepsize * (
            stat2 / nchains - suffStat["statphi2"]
        )
        suffStat["statphi3"] = suffStat["statphi3"] + stepsize * (
            stat3 / nchains - suffStat["statphi3"]
        )

        # Estimate mean_phi (fixed effects with covariates)
        mean_phi = _estimate_mean_phi(phiM, Uargs)

        # Update omega using the correct SAEM formula from R saemix
        # omega = statphi2/N + e1_phi'*e1_phi/N - statphi1'*e1_phi/N - e1_phi'*statphi1/N
        # where e1_phi is mean_phi[:, ind_eta] (covariate effects on random effect parameters)
        omega_full = np.zeros_like(varList["omega"])

        if nb_etas > 0:
            # e1_phi: the mean phi values for random effect parameters (N x nb_etas)
            e1_phi = mean_phi[:, ind_eta]

            # Compute omega using the full formula matching R implementation
            # This accounts for covariate effects on random effect parameters
            omega_eta = (
                suffStat["statphi2"] / N
                + (e1_phi.T @ e1_phi) / N
                - (suffStat["statphi1"].T @ e1_phi) / N
                - (e1_phi.T @ suffStat["statphi1"]) / N
            )

            omega_full[np.ix_(ind_eta, ind_eta)] = omega_eta

        # Ensure non-negative values (numerical safety)
        omega_full = np.where(omega_full < 0, 1e-6, omega_full)

        # Apply covariance model structure (which elements to estimate)
        omega_new = np.zeros_like(varList["omega"])
        indest_omega = Uargs.get("indest_omega", None)
        if indest_omega is not None:
            omega_new[indest_omega] = omega_full[indest_omega]
        else:
            omega_new = omega_full

        # Apply simulated annealing to diagonal elements during burn-in phase
        # This matches the R saemix implementation
        i0_omega2 = Uargs.get(
            "i0_omega2", np.array([], dtype=int)
        )  # params without IIV
        i1_omega2 = Uargs.get("i1_omega2", ind_eta)  # params with IIV

        # Ensure diag_omega is writable (may be a view from initialization)
        if not varList["diag_omega"].flags.writeable:
            varList["diag_omega"] = varList["diag_omega"].copy()

        if kiter <= opt["nbiter_sa"]:
            # Simulated annealing phase
            diag_omega_full = np.diag(omega_new)

            if len(i1_omega2) > 0:
                vec1 = diag_omega_full[i1_omega2]
                vec2 = varList["diag_omega"][i1_omega2] * opt["alpha1_sa"]
                # Use the larger of current SA value or new estimate
                idx = (vec1 < vec2).astype(int)
                varList["diag_omega"][i1_omega2] = idx * vec2 + (1 - idx) * vec1

            if len(i0_omega2) > 0:
                # Decay parameters without IIV
                varList["diag_omega"][i0_omega2] = varList["diag_omega"][
                    i0_omega2
                ] * opt.get("alpha0_sa", 0.97)
        else:
            # After SA phase, use the computed diagonal
            varList["diag_omega"] = np.diag(omega_new)

        # Update omega with the (possibly SA-modified) diagonal
        omega_new = (
            omega_new - np.diag(np.diag(omega_new)) + np.diag(varList["diag_omega"])
        )

        # Ensure positive definiteness
        omega_new = _ensure_positive_definite(omega_new)

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


def _ensure_positive_definite(omega, min_eigenvalue=1e-8):
    """
    Ensure a matrix is positive definite by correcting small/negative eigenvalues.

    Parameters
    ----------
    omega : np.ndarray
        Covariance matrix to check/correct
    min_eigenvalue : float
        Minimum allowed eigenvalue

    Returns
    -------
    np.ndarray
        Positive definite covariance matrix
    """
    # Ensure symmetry
    omega = (omega + omega.T) / 2

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(omega)

        if np.any(eigenvalues < min_eigenvalue):
            # Correct small/negative eigenvalues
            eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
            # Reconstruct matrix
            omega = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            # Ensure symmetry after reconstruction
            omega = (omega + omega.T) / 2
    except np.linalg.LinAlgError:
        # If eigendecomposition fails, add small diagonal
        omega = omega + min_eigenvalue * np.eye(omega.shape[0])

    return omega
