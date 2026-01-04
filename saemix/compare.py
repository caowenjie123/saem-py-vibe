"""
Model Comparison Module

This module provides functions for comparing multiple fitted SAEM models
using information criteria (AIC, BIC, BIC_cov).
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union

from saemix.results import SaemixObject


def compare_saemix(
    *models: SaemixObject, method: str = "is", names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple fitted models using information criteria.

    Parameters
    ----------
    *models : SaemixObject
        Multiple fitted model objects to compare
    method : str
        Likelihood calculation method: 'is' (importance sampling) or 'gq' (Gaussian quadrature)
        Default is 'is'
    names : list of str, optional
        Model names for the comparison table. If not provided, models are named
        'Model 1', 'Model 2', etc.

    Returns
    -------
    pd.DataFrame
        Comparison results table with columns:
        - model: Model name
        - npar: Number of estimated parameters
        - ll: Log-likelihood
        - AIC: Akaike Information Criterion
        - BIC: Bayesian Information Criterion
        - BIC_cov: Covariate-specific BIC (using total observations)

    Raises
    ------
    ValueError
        If fewer than 2 models are provided
        If models have different data (different n_subjects or n_observations)
        If method is not one of 'is', 'gq'

    Notes
    -----
    Information criteria are computed as:
    - AIC = -2 * LL + 2 * k
    - BIC = -2 * LL + log(N) * k  (N = number of subjects)
    - BIC_cov = -2 * LL + log(n_total_obs) * k

    where k is the number of estimated parameters.

    Examples
    --------
    >>> result1 = saemix(model1, data, control)
    >>> result2 = saemix(model2, data, control)
    >>> comparison = compare_saemix(result1, result2, method='is')
    >>> print(comparison)
    """
    # Validate inputs
    if len(models) < 2:
        raise ValueError("At least 2 models are required for comparison")

    valid_methods = ["is", "gq"]
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    # Validate that all models have the same data dimensions
    ref_n_subjects = models[0].data.n_subjects
    ref_n_obs = models[0].data.n_total_obs

    for i, model in enumerate(models[1:], start=2):
        if model.data.n_subjects != ref_n_subjects:
            raise ValueError(
                f"Model {i} has {model.data.n_subjects} subjects, "
                f"but Model 1 has {ref_n_subjects} subjects. "
                "All models must have the same data."
            )
        if model.data.n_total_obs != ref_n_obs:
            raise ValueError(
                f"Model {i} has {model.data.n_total_obs} observations, "
                f"but Model 1 has {ref_n_obs} observations. "
                "All models must have the same data."
            )

    # Generate model names if not provided
    if names is None:
        names = [f"Model {i+1}" for i in range(len(models))]
    elif len(names) != len(models):
        raise ValueError(
            f"Number of names ({len(names)}) must match number of models ({len(models)})"
        )

    # Compute likelihood for each model if not already computed
    from saemix.algorithm.likelihood import llis_saemix, llgq_saemix

    results = []

    for i, saemix_obj in enumerate(models):
        # Ensure likelihood is computed
        if method == "is":
            if saemix_obj.results.ll_is is None:
                try:
                    llis_saemix(saemix_obj)
                except Exception as e:
                    raise ValueError(
                        f"Failed to compute IS likelihood for {names[i]}: {e}"
                    )
            ll = saemix_obj.results.ll_is
            aic = saemix_obj.results.aic_is
            bic = saemix_obj.results.bic_is
        else:  # method == 'gq'
            if saemix_obj.results.ll_gq is None:
                try:
                    llgq_saemix(saemix_obj)
                except Exception as e:
                    raise ValueError(
                        f"Failed to compute GQ likelihood for {names[i]}: {e}"
                    )
            ll = saemix_obj.results.ll_gq
            aic = saemix_obj.results.aic_gq
            bic = saemix_obj.results.bic_gq

        # Get number of parameters
        npar = _compute_npar(saemix_obj)

        # Compute BIC_cov (using total observations instead of subjects)
        n_total_obs = saemix_obj.data.n_total_obs
        bic_cov = -2 * ll + np.log(n_total_obs) * npar

        results.append(
            {
                "model": names[i],
                "npar": npar,
                "ll": ll,
                "AIC": aic,
                "BIC": bic,
                "BIC_cov": bic_cov,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by AIC (best model first)
    df = df.sort_values("AIC").reset_index(drop=True)

    return df


def _compute_npar(saemix_obj: SaemixObject) -> int:
    """
    Compute the number of estimated parameters for a model.

    Parameters
    ----------
    saemix_obj : SaemixObject
        Fitted SAEM result object

    Returns
    -------
    int
        Number of estimated parameters
    """
    # Use cached value if available
    if saemix_obj.results.npar_est is not None:
        return saemix_obj.results.npar_est

    model = saemix_obj.model

    # Count fixed effects
    betaest_model = getattr(model, "betaest_model", None)
    if betaest_model is None:
        betaest_model = np.ones((1, model.n_parameters))

    covariate_estim = betaest_model.copy()
    covariate_estim[0, :] = model.fixed_estim
    n_fixed = int(np.sum(covariate_estim))

    # Count omega parameters (upper triangle of covariance matrix)
    omega_upper = np.triu(model.covariance_model, k=0)
    n_omega = int(np.sum(omega_upper != 0))

    # Count residual error parameters
    n_res = 0
    if model.modeltype == "structural":
        for em in model.error_model:
            if em == "combined":
                n_res += 2
            else:
                n_res += 1

    npar = n_fixed + n_omega + n_res

    # Cache the result
    saemix_obj.results.npar_est = npar

    return npar


def aic(saemix_obj: SaemixObject, method: str = "is") -> float:
    """
    Get the AIC value for a fitted model.

    Parameters
    ----------
    saemix_obj : SaemixObject
        Fitted SAEM result object
    method : str
        Likelihood method: 'is' or 'gq'

    Returns
    -------
    float
        AIC value
    """
    if method == "is":
        if saemix_obj.results.aic_is is None:
            from saemix.algorithm.likelihood import llis_saemix

            llis_saemix(saemix_obj)
        return saemix_obj.results.aic_is
    else:
        if saemix_obj.results.aic_gq is None:
            from saemix.algorithm.likelihood import llgq_saemix

            llgq_saemix(saemix_obj)
        return saemix_obj.results.aic_gq


def bic(saemix_obj: SaemixObject, method: str = "is") -> float:
    """
    Get the BIC value for a fitted model.

    Parameters
    ----------
    saemix_obj : SaemixObject
        Fitted SAEM result object
    method : str
        Likelihood method: 'is' or 'gq'

    Returns
    -------
    float
        BIC value
    """
    if method == "is":
        if saemix_obj.results.bic_is is None:
            from saemix.algorithm.likelihood import llis_saemix

            llis_saemix(saemix_obj)
        return saemix_obj.results.bic_is
    else:
        if saemix_obj.results.bic_gq is None:
            from saemix.algorithm.likelihood import llgq_saemix

            llgq_saemix(saemix_obj)
        return saemix_obj.results.bic_gq


def loglik(saemix_obj: SaemixObject, method: str = "is") -> float:
    """
    Get the log-likelihood value for a fitted model.

    Parameters
    ----------
    saemix_obj : SaemixObject
        Fitted SAEM result object
    method : str
        Likelihood method: 'is' or 'gq'

    Returns
    -------
    float
        Log-likelihood value
    """
    if method == "is":
        if saemix_obj.results.ll_is is None:
            from saemix.algorithm.likelihood import llis_saemix

            llis_saemix(saemix_obj)
        return saemix_obj.results.ll_is
    else:
        if saemix_obj.results.ll_gq is None:
            from saemix.algorithm.likelihood import llgq_saemix

            llgq_saemix(saemix_obj)
        return saemix_obj.results.ll_gq
