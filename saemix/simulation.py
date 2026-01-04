"""
Simulation Module

This module provides functions for simulating data from fitted SAEM models,
supporting both continuous and discrete response types.

Feature: saemix-python-enhancement
Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
"""

import numpy as np
import pandas as pd
from typing import Optional, Callable, Union, List
from scipy.stats import norm

from saemix.results import SaemixObject
from saemix.utils import transphi, cutoff
from saemix.algorithm.map_estimation import error_function


def simulate_saemix(
    saemix_object: SaemixObject,
    nsim: int = 1000,
    seed: Optional[int] = None,
    predictions: bool = True,
    res_var: bool = True,
) -> pd.DataFrame:
    """
    Simulate data from a fitted SAEM model.

    Generates simulated observations using the estimated population parameters
    and random effects distribution.

    Parameters
    ----------
    saemix_object : SaemixObject
        Fitted SAEM result object
    nsim : int
        Number of simulation replicates (default: 1000)
    seed : int, optional
        Random seed for reproducibility
    predictions : bool
        If True, include population and individual predictions in output
    res_var : bool
        If True, add residual variability to simulated observations

    Returns
    -------
    pd.DataFrame
        Simulated data with columns:
        - sim: Simulation replicate number (1 to nsim)
        - id: Subject ID
        - time: Time/predictor values
        - ysim: Simulated observations
        - ppred: Population predictions (if predictions=True)
        - ipred: Individual predictions (if predictions=True)

    Raises
    ------
    ValueError
        If nsim < 1
        If SaemixObject has not been fitted

    Notes
    -----
    The simulation process:
    1. For each replicate, sample random effects from N(0, Omega)
    2. Compute individual parameters: phi_i = mu + eta_i
    3. Compute predictions using the structural model
    4. Add residual error if res_var=True

    Examples
    --------
    >>> result = saemix(model, data, control)
    >>> sim_data = simulate_saemix(result, nsim=100, seed=42)
    >>> print(sim_data.head())
    """
    # Validate inputs
    if nsim < 1:
        raise ValueError("nsim must be >= 1")

    if saemix_object.results.mean_phi is None:
        raise ValueError("SaemixObject has not been fitted. Run saemix() first.")

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Extract model components
    model = saemix_object.model
    data = saemix_object.data
    results = saemix_object.results

    # Get population parameters
    mean_phi = np.atleast_2d(results.mean_phi)
    if mean_phi.shape[0] > 1:
        # Use first row as population mean
        pop_phi = mean_phi[0, :]
    else:
        pop_phi = mean_phi.flatten()

    # Get omega (random effects covariance)
    omega = results.omega
    n_parameters = model.n_parameters

    # Get data structure
    xind = data.data[data.name_predictors].values
    index = data.data["index"].values
    id_col = data.data[data.name_group].values
    time_col = data.data[data.name_predictors[0]].values
    n_obs = len(index)
    n_subjects = data.n_subjects

    # Get residual error parameters
    respar = results.respar if results.respar is not None else model.error_init

    # Get indices of parameters with random effects
    ind_eta = model.indx_omega

    # Prepare output lists
    sim_results = []

    for sim_idx in range(1, nsim + 1):
        # Sample random effects for all subjects
        if len(ind_eta) > 0:
            omega_sub = omega[np.ix_(ind_eta, ind_eta)]
            try:
                eta = np.random.multivariate_normal(
                    np.zeros(len(ind_eta)), omega_sub, size=n_subjects
                )
            except np.linalg.LinAlgError:
                # If omega is singular, use diagonal
                eta = np.random.normal(
                    0, np.sqrt(np.diag(omega_sub)), size=(n_subjects, len(ind_eta))
                )
        else:
            eta = np.zeros((n_subjects, 0))

        # Compute individual parameters (phi space)
        phi_sim = np.tile(pop_phi, (n_subjects, 1))
        if len(ind_eta) > 0:
            phi_sim[:, ind_eta] += eta

        # Transform to psi space
        psi_sim = transphi(phi_sim, model.transform_par)

        # Compute predictions
        f = model.model(psi_sim, index, xind)
        ipred = f.copy()

        # Compute population predictions (using pop_phi for all subjects)
        pop_psi = transphi(pop_phi.reshape(1, -1), model.transform_par)
        pop_psi_expanded = np.tile(pop_psi, (n_subjects, 1))
        ppred = model.model(pop_psi_expanded, index, xind)

        # Add residual variability
        if res_var and model.modeltype == "structural":
            ytype = data.data["ytype"].values if "ytype" in data.data.columns else None
            g = error_function(f, respar, model.error_model, ytype)
            epsilon = np.random.normal(0, 1, n_obs)
            ysim = f + g * epsilon
        else:
            ysim = f.copy()

        # Build result for this simulation
        for i in range(n_obs):
            row = {
                "sim": sim_idx,
                "id": id_col[i],
                "time": time_col[i],
                "ysim": ysim[i],
            }
            if predictions:
                row["ppred"] = ppred[i]
                row["ipred"] = ipred[i]

            sim_results.append(row)

    # Create DataFrame
    df = pd.DataFrame(sim_results)

    return df


def simulate_discrete_saemix(
    saemix_object: SaemixObject,
    simulate_function: Callable,
    nsim: int = 1000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate discrete response data from a fitted SAEM model.

    Uses a user-provided simulation function to generate discrete outcomes
    based on the model predictions.

    Parameters
    ----------
    saemix_object : SaemixObject
        Fitted SAEM result object
    simulate_function : callable
        Function to generate discrete outcomes. Should accept:
        - psi: Individual parameters (n_subjects, n_parameters)
        - id: Subject indices for each observation
        - xidep: Predictor values
        And return simulated discrete outcomes.
    nsim : int
        Number of simulation replicates (default: 1000)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Simulated data with columns:
        - sim: Simulation replicate number (1 to nsim)
        - id: Subject ID
        - time: Time/predictor values
        - ysim: Simulated discrete outcomes

    Raises
    ------
    ValueError
        If nsim < 1
        If SaemixObject has not been fitted
        If simulate_function is not callable

    Notes
    -----
    The simulate_function should handle the discrete nature of the response.
    For example, for binary outcomes, it might use a logistic function
    and sample from a Bernoulli distribution.

    Examples
    --------
    >>> def simulate_binary(psi, id, xidep):
    ...     prob = 1 / (1 + np.exp(-psi[id, 0] * xidep[:, 0] - psi[id, 1]))
    ...     return np.random.binomial(1, prob)
    >>>
    >>> result = saemix(model, data, control)
    >>> sim_data = simulate_discrete_saemix(result, simulate_binary, nsim=100)
    """
    # Validate inputs
    if nsim < 1:
        raise ValueError("nsim must be >= 1")

    if saemix_object.results.mean_phi is None:
        raise ValueError("SaemixObject has not been fitted. Run saemix() first.")

    if not callable(simulate_function):
        raise ValueError("simulate_function must be callable")

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Extract model components
    model = saemix_object.model
    data = saemix_object.data
    results = saemix_object.results

    # Get population parameters
    mean_phi = np.atleast_2d(results.mean_phi)
    if mean_phi.shape[0] > 1:
        pop_phi = mean_phi[0, :]
    else:
        pop_phi = mean_phi.flatten()

    # Get omega
    omega = results.omega
    n_parameters = model.n_parameters

    # Get data structure
    xind = data.data[data.name_predictors].values
    index = data.data["index"].values
    id_col = data.data[data.name_group].values
    time_col = data.data[data.name_predictors[0]].values
    n_obs = len(index)
    n_subjects = data.n_subjects

    # Get indices of parameters with random effects
    ind_eta = model.indx_omega

    # Prepare output lists
    sim_results = []

    for sim_idx in range(1, nsim + 1):
        # Sample random effects
        if len(ind_eta) > 0:
            omega_sub = omega[np.ix_(ind_eta, ind_eta)]
            try:
                eta = np.random.multivariate_normal(
                    np.zeros(len(ind_eta)), omega_sub, size=n_subjects
                )
            except np.linalg.LinAlgError:
                eta = np.random.normal(
                    0, np.sqrt(np.diag(omega_sub)), size=(n_subjects, len(ind_eta))
                )
        else:
            eta = np.zeros((n_subjects, 0))

        # Compute individual parameters
        phi_sim = np.tile(pop_phi, (n_subjects, 1))
        if len(ind_eta) > 0:
            phi_sim[:, ind_eta] += eta

        # Transform to psi space
        psi_sim = transphi(phi_sim, model.transform_par)

        # Generate discrete outcomes using user function
        ysim = simulate_function(psi_sim, index, xind)

        # Build result for this simulation
        for i in range(n_obs):
            row = {
                "sim": sim_idx,
                "id": id_col[i],
                "time": time_col[i],
                "ysim": ysim[i],
            }
            sim_results.append(row)

    # Create DataFrame
    df = pd.DataFrame(sim_results)

    return df


def simulate_with_uncertainty(
    saemix_object: SaemixObject,
    nsim: int = 1000,
    seed: Optional[int] = None,
    predictions: bool = True,
    res_var: bool = True,
    parameter_uncertainty: bool = True,
) -> pd.DataFrame:
    """
    Simulate data with parameter uncertainty propagation.

    In addition to random effects variability, this function also samples
    from the uncertainty distribution of the population parameters.

    Parameters
    ----------
    saemix_object : SaemixObject
        Fitted SAEM result object
    nsim : int
        Number of simulation replicates (default: 1000)
    seed : int, optional
        Random seed for reproducibility
    predictions : bool
        If True, include population and individual predictions in output
    res_var : bool
        If True, add residual variability to simulated observations
    parameter_uncertainty : bool
        If True, sample population parameters from their uncertainty distribution

    Returns
    -------
    pd.DataFrame
        Simulated data with columns similar to simulate_saemix

    Notes
    -----
    When parameter_uncertainty=True, the population parameters are sampled
    from a multivariate normal distribution using the Fisher Information Matrix
    to estimate the covariance of the parameter estimates.
    """
    # Validate inputs
    if nsim < 1:
        raise ValueError("nsim must be >= 1")

    if saemix_object.results.mean_phi is None:
        raise ValueError("SaemixObject has not been fitted. Run saemix() first.")

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Extract model components
    model = saemix_object.model
    data = saemix_object.data
    results = saemix_object.results

    # Get population parameters
    mean_phi = np.atleast_2d(results.mean_phi)
    if mean_phi.shape[0] > 1:
        pop_phi = mean_phi[0, :]
    else:
        pop_phi = mean_phi.flatten()

    n_parameters = model.n_parameters

    # Get parameter covariance from FIM if available
    if parameter_uncertainty and results.fim is not None:
        try:
            param_cov = np.linalg.inv(results.fim)[:n_parameters, :n_parameters]
        except np.linalg.LinAlgError:
            param_cov = None
    else:
        param_cov = None

    # Get omega
    omega = results.omega

    # Get data structure
    xind = data.data[data.name_predictors].values
    index = data.data["index"].values
    id_col = data.data[data.name_group].values
    time_col = data.data[data.name_predictors[0]].values
    n_obs = len(index)
    n_subjects = data.n_subjects

    # Get residual error parameters
    respar = results.respar if results.respar is not None else model.error_init

    # Get indices of parameters with random effects
    ind_eta = model.indx_omega

    # Prepare output lists
    sim_results = []

    for sim_idx in range(1, nsim + 1):
        # Sample population parameters if uncertainty is enabled
        if parameter_uncertainty and param_cov is not None:
            try:
                pop_phi_sim = np.random.multivariate_normal(pop_phi, param_cov)
            except np.linalg.LinAlgError:
                pop_phi_sim = pop_phi.copy()
        else:
            pop_phi_sim = pop_phi.copy()

        # Sample random effects
        if len(ind_eta) > 0:
            omega_sub = omega[np.ix_(ind_eta, ind_eta)]
            try:
                eta = np.random.multivariate_normal(
                    np.zeros(len(ind_eta)), omega_sub, size=n_subjects
                )
            except np.linalg.LinAlgError:
                eta = np.random.normal(
                    0, np.sqrt(np.diag(omega_sub)), size=(n_subjects, len(ind_eta))
                )
        else:
            eta = np.zeros((n_subjects, 0))

        # Compute individual parameters
        phi_sim = np.tile(pop_phi_sim, (n_subjects, 1))
        if len(ind_eta) > 0:
            phi_sim[:, ind_eta] += eta

        # Transform to psi space
        psi_sim = transphi(phi_sim, model.transform_par)

        # Compute predictions
        f = model.model(psi_sim, index, xind)
        ipred = f.copy()

        # Compute population predictions
        pop_psi = transphi(pop_phi_sim.reshape(1, -1), model.transform_par)
        pop_psi_expanded = np.tile(pop_psi, (n_subjects, 1))
        ppred = model.model(pop_psi_expanded, index, xind)

        # Add residual variability
        if res_var and model.modeltype == "structural":
            ytype = data.data["ytype"].values if "ytype" in data.data.columns else None
            g = error_function(f, respar, model.error_model, ytype)
            epsilon = np.random.normal(0, 1, n_obs)
            ysim = f + g * epsilon
        else:
            ysim = f.copy()

        # Build result for this simulation
        for i in range(n_obs):
            row = {
                "sim": sim_idx,
                "id": id_col[i],
                "time": time_col[i],
                "ysim": ysim[i],
            }
            if predictions:
                row["ppred"] = ppred[i]
                row["ipred"] = ipred[i]

            sim_results.append(row)

    # Create DataFrame
    df = pd.DataFrame(sim_results)

    return df
