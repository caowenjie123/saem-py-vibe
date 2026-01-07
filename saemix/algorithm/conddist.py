"""
Conditional Distribution Estimation Module

This module implements MCMC-based conditional distribution estimation for
individual parameters in nonlinear mixed effects models.

The main function `conddist_saemix` uses Metropolis-Hastings sampling to
estimate the conditional distribution p(phi_i | y_i, theta) for each subject.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from scipy.optimize import minimize

from saemix.utils import transphi, cutoff


def conddist_saemix(
    saemix_object: "SaemixObject",
    nsamp: int = 1,
    max_iter: Optional[int] = None,
    nburn: int = 0,
    nchains: int = 1,
    plot: bool = False,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> "SaemixObject":
    """
    Estimate the conditional distribution of individual parameters using MCMC.

    Implements Metropolis-Hastings sampling from p(phi_i | y_i, theta).

    Parameters
    ----------
    saemix_object : SaemixObject
        A fitted SAEM result object
    nsamp : int
        Number of samples per subject (default: 1)
    max_iter : int, optional
        Maximum MCMC iterations, default is nsamp * 10
    nburn : int
        Number of burn-in iterations (default: 0)
    nchains : int
        Number of MCMC chains (default: 1)
    plot : bool
        Whether to display convergence diagnostic plots (default: False)
    seed : int, optional
        Random seed for reproducibility (deprecated, use rng instead)
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility. Priority: rng > seed > saemix_object.control.rng

    Returns
    -------
    SaemixObject
        Updated result object containing:
        - cond_mean_phi: Conditional mean (n_subjects, n_parameters)
        - cond_var_phi: Conditional variance (n_subjects, n_parameters) with all values >= 0
        - cond_shrinkage: Shrinkage estimates (n_parameters,)
        - phi_samp: Samples (n_subjects, nsamp, n_parameters)

    Raises
    ------
    ValueError
        If saemix_object has not been fitted
        If nsamp < 1
        If max_iter < nsamp
    """
    # Validate inputs
    if saemix_object.results.omega is None:
        raise ValueError("SaemixObject has not been fitted. Run SAEM algorithm first.")

    if nsamp < 1:
        raise ValueError("nsamp must be >= 1")

    if max_iter is None:
        max_iter = nsamp * 10

    if max_iter < nsamp:
        raise ValueError("max_iter must be >= nsamp")

    # Determine RNG to use - priority: rng > seed > saemix_object.control.rng > new default
    if rng is not None:
        _rng = rng
    elif seed is not None:
        _rng = np.random.default_rng(seed)
    elif hasattr(saemix_object, "control") and saemix_object.control is not None:
        _rng = saemix_object.control.get("rng", None)
        if _rng is None:
            _rng = np.random.default_rng()
    else:
        _rng = np.random.default_rng()

    # Extract necessary components
    model = saemix_object.model
    data = saemix_object.data
    results = saemix_object.results

    # Get dimensions
    N = data.n_subjects
    nb_parameters = model.n_parameters
    ind_eta = model.indx_omega  # Indices of parameters with random effects
    nb_etas = len(ind_eta)

    # Get omega matrix for random effects
    omega = results.omega
    omega_eta = omega[np.ix_(ind_eta, ind_eta)] if nb_etas > 0 else np.zeros((0, 0))

    # Ensure omega_eta is positive definite
    if nb_etas > 0:
        domega = cutoff(np.diag(omega_eta), 1e-10)
        omega_eta = omega_eta - np.diag(np.diag(omega_eta)) + np.diag(domega)
        try:
            chol_omega = np.linalg.cholesky(omega_eta)
            inv_omega = np.linalg.inv(omega_eta)
        except np.linalg.LinAlgError:
            chol_omega = np.eye(nb_etas)
            inv_omega = np.eye(nb_etas)
    else:
        chol_omega = np.zeros((0, 0))
        inv_omega = np.zeros((0, 0))

    # Get residual error parameters
    pres = results.respar if results.respar is not None else model.error_init

    # Initialize storage for samples
    phi_samp = np.zeros((N, nsamp, nb_parameters))

    # Get data arrays
    id_col = data.data[data.name_group].values
    xind = data.data[data.name_predictors].values
    yobs = data.data[data.name_response].values
    id_list = np.unique(id_col)

    # Get mean_phi
    mean_phi = results.mean_phi
    if mean_phi is None:
        raise ValueError("mean_phi not found. Run SAEM algorithm first.")

    # Initialize phi from MAP estimates if available, otherwise from mean_phi
    if results.map_phi is not None:
        phi_current = np.array(results.map_phi).copy()
        if phi_current.ndim == 1:
            phi_current = phi_current.reshape(N, -1)
    else:
        phi_current = mean_phi.copy()

    # Run MCMC for each subject
    all_samples = []  # Store all samples for each subject

    for i in range(N):
        isuj = id_list[i]
        mask = id_col == isuj
        xi = xind[mask]
        yi = yobs[mask]
        idi = np.zeros(len(yi), dtype=int)

        # Get subject-specific mean
        mean_phi_i = mean_phi[i, :]

        # Initialize chains
        chain_samples = []

        for chain in range(nchains):
            # Initialize from current phi with small perturbation for multiple chains
            if chain == 0:
                phi_i = phi_current[i, :].copy()
            else:
                phi_i = phi_current[i, :].copy()
                if nb_etas > 0:
                    phi_i[ind_eta] += _rng.standard_normal(nb_etas) @ chol_omega.T * 0.1

            # Compute initial log-posterior
            log_post = _compute_log_posterior(
                phi_i, mean_phi_i, ind_eta, inv_omega, idi, xi, yi, model, pres, data
            )

            # Adaptive proposal scale
            proposal_scale = np.ones(nb_etas) if nb_etas > 0 else np.array([])

            # MCMC iterations
            samples_i = []
            n_accepted = 0

            for iter_idx in range(max_iter + nburn):
                if nb_etas > 0:
                    # Propose new phi
                    phi_prop = phi_i.copy()
                    eta_current = phi_i[ind_eta] - mean_phi_i[ind_eta]
                    eta_prop = eta_current + _rng.standard_normal(
                        nb_etas
                    ) * proposal_scale * np.sqrt(np.diag(omega_eta))
                    phi_prop[ind_eta] = mean_phi_i[ind_eta] + eta_prop

                    # Compute log-posterior for proposal
                    log_post_prop = _compute_log_posterior(
                        phi_prop,
                        mean_phi_i,
                        ind_eta,
                        inv_omega,
                        idi,
                        xi,
                        yi,
                        model,
                        pres,
                        data,
                    )

                    # Metropolis-Hastings acceptance
                    log_alpha = log_post_prop - log_post

                    if np.log(_rng.random()) < log_alpha:
                        phi_i = phi_prop
                        log_post = log_post_prop
                        n_accepted += 1

                    # Adapt proposal scale during burn-in
                    if iter_idx < nburn and iter_idx > 0 and iter_idx % 50 == 0:
                        accept_rate = n_accepted / (iter_idx + 1)
                        if accept_rate < 0.2:
                            proposal_scale *= 0.8
                        elif accept_rate > 0.5:
                            proposal_scale *= 1.2
                        proposal_scale = np.clip(proposal_scale, 0.01, 10.0)

                # Store sample after burn-in
                if iter_idx >= nburn:
                    samples_i.append(phi_i.copy())

            chain_samples.append(np.array(samples_i))

        # Combine chains and thin to get nsamp samples
        all_chain_samples = np.concatenate(chain_samples, axis=0)
        n_total_samples = len(all_chain_samples)

        # Select nsamp samples evenly spaced
        if n_total_samples >= nsamp:
            indices = np.linspace(0, n_total_samples - 1, nsamp, dtype=int)
            phi_samp[i, :, :] = all_chain_samples[indices]
        else:
            # If not enough samples, repeat the last one
            phi_samp[i, :n_total_samples, :] = all_chain_samples
            phi_samp[i, n_total_samples:, :] = all_chain_samples[-1]

        all_samples.append(all_chain_samples)

    # Compute conditional mean and variance
    cond_mean_phi = np.mean(phi_samp, axis=1)  # (N, nb_parameters)
    cond_var_phi = np.var(phi_samp, axis=1)  # (N, nb_parameters)

    # Ensure variance is non-negative
    cond_var_phi = np.maximum(cond_var_phi, 0.0)

    # Compute shrinkage
    cond_shrinkage = _compute_shrinkage(cond_mean_phi, omega, ind_eta, nb_parameters)

    # Compute conditional mean in psi space
    cond_mean_psi = transphi(cond_mean_phi, model.transform_par)

    # Update results
    results.cond_mean_phi = cond_mean_phi
    results.cond_var_phi = cond_var_phi
    results.cond_shrinkage = cond_shrinkage
    results.phi_samp = phi_samp
    results.cond_mean_psi = cond_mean_psi

    # Plot convergence diagnostics if requested
    if plot:
        _plot_convergence_diagnostics(all_samples, model.name_modpar, ind_eta)

    return saemix_object


def _compute_log_posterior(
    phi: np.ndarray,
    mean_phi: np.ndarray,
    ind_eta: np.ndarray,
    inv_omega: np.ndarray,
    idi: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
    model: "SaemixModel",
    pres: np.ndarray,
    data: "SaemixData",
) -> float:
    """
    Compute the log-posterior for a given phi.

    log p(phi | y, theta) = log p(y | phi) + log p(phi | theta) + const
    """
    from saemix.algorithm.map_estimation import error_function

    nb_etas = len(ind_eta)

    # Compute log-likelihood p(y | phi)
    phi_2d = phi.reshape(1, -1)
    psi = transphi(phi_2d, model.transform_par)

    try:
        fpred = model.model(psi, idi, xi)

        # Handle exponential error model
        error_model = model.error_model
        if "exponential" in error_model:
            fpred = np.log(cutoff(fpred))

        gpred = error_function(fpred, pres, error_model, None)

        # Log-likelihood (Gaussian)
        log_lik = -0.5 * np.sum(((yi - fpred) / gpred) ** 2) - np.sum(np.log(gpred))
    except Exception:
        return -np.inf

    # Compute log-prior p(phi | theta)
    if nb_etas > 0:
        eta = phi[ind_eta] - mean_phi[ind_eta]
        log_prior = -0.5 * eta @ inv_omega @ eta
    else:
        log_prior = 0.0

    return log_lik + log_prior


def _compute_shrinkage(
    cond_mean_phi: np.ndarray,
    omega: np.ndarray,
    ind_eta: np.ndarray,
    nb_parameters: int,
) -> np.ndarray:
    """
    Compute shrinkage estimates for each parameter.

    Shrinkage = 1 - var(cond_mean) / var(population)

    Returns values between 0 and 1.
    """
    shrinkage = np.zeros(nb_parameters)

    for j in range(nb_parameters):
        if j in ind_eta:
            # Variance of conditional means
            var_cond_mean = np.var(cond_mean_phi[:, j])
            # Population variance (from omega)
            var_pop = omega[j, j]

            if var_pop > 0:
                shrinkage[j] = 1.0 - var_cond_mean / var_pop
            else:
                shrinkage[j] = 1.0

            # Clip to [0, 1]
            shrinkage[j] = np.clip(shrinkage[j], 0.0, 1.0)
        else:
            # No random effect, shrinkage is 1 (fully shrunk to population)
            shrinkage[j] = 1.0

    return shrinkage


def _plot_convergence_diagnostics(
    all_samples: list, param_names: list, ind_eta: np.ndarray
) -> None:
    """
    Plot convergence diagnostics for MCMC samples.
    """
    try:
        import matplotlib.pyplot as plt

        n_subjects = len(all_samples)
        n_params = len(ind_eta)

        if n_params == 0:
            print("No random effects to plot.")
            return

        # Plot trace for first few subjects
        n_plot = min(3, n_subjects)

        fig, axes = plt.subplots(n_plot, n_params, figsize=(4 * n_params, 3 * n_plot))
        if n_plot == 1:
            axes = axes.reshape(1, -1)
        if n_params == 1:
            axes = axes.reshape(-1, 1)

        for i in range(n_plot):
            samples = all_samples[i]
            for j, idx in enumerate(ind_eta):
                ax = axes[i, j]
                ax.plot(samples[:, idx], alpha=0.7)
                ax.set_xlabel("Iteration")
                if param_names and idx < len(param_names):
                    ax.set_ylabel(param_names[idx])
                else:
                    ax.set_ylabel(f"Parameter {idx}")
                if i == 0:
                    ax.set_title(f"Subject {i+1}")

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("matplotlib not available for plotting")


def compute_gelman_rubin(chains: list) -> np.ndarray:
    """
    Compute Gelman-Rubin convergence diagnostic (R-hat).

    Parameters
    ----------
    chains : list of np.ndarray
        List of MCMC chains, each of shape (n_samples, n_params)

    Returns
    -------
    np.ndarray
        R-hat values for each parameter
    """
    if len(chains) < 2:
        return np.ones(chains[0].shape[1])

    n_chains = len(chains)
    n_samples = chains[0].shape[0]
    n_params = chains[0].shape[1]

    # Stack chains
    all_chains = np.array(chains)  # (n_chains, n_samples, n_params)

    # Chain means
    chain_means = np.mean(all_chains, axis=1)  # (n_chains, n_params)

    # Overall mean
    overall_mean = np.mean(chain_means, axis=0)  # (n_params,)

    # Between-chain variance
    B = n_samples * np.var(chain_means, axis=0, ddof=1)  # (n_params,)

    # Within-chain variance
    chain_vars = np.var(all_chains, axis=1, ddof=1)  # (n_chains, n_params)
    W = np.mean(chain_vars, axis=0)  # (n_params,)

    # Pooled variance estimate
    var_hat = (1 - 1 / n_samples) * W + (1 / n_samples) * B

    # R-hat
    R_hat = np.sqrt(var_hat / W)

    return R_hat
