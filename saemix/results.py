"""
Results Module

This module contains the SaemixRes and SaemixObject classes for storing
and accessing SAEM estimation results.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


class SaemixRes:
    """
    Results class for storing SAEM estimation outputs.

    Attributes
    ----------
    fixed_effects : np.ndarray
        Estimated fixed effect parameters
    omega : np.ndarray
        Estimated variance-covariance matrix of random effects
    respar : np.ndarray
        Estimated residual error parameters
    ll : float
        Log-likelihood value
    aic : float
        Akaike Information Criterion
    bic : float
        Bayesian Information Criterion
    ll_is : float
        Log-likelihood computed by importance sampling
    aic_is : float
        AIC computed using IS likelihood
    bic_is : float
        BIC computed using IS likelihood
    ll_gq : float
        Log-likelihood computed by Gaussian quadrature
    aic_gq : float
        AIC computed using GQ likelihood
    bic_gq : float
        BIC computed using GQ likelihood
    npar_est : int
        Number of estimated parameters
    mean_phi : np.ndarray
        Population mean parameters (phi space)
    map_phi : np.ndarray
        MAP estimates of individual parameters (phi space)
    map_psi : np.ndarray
        MAP estimates of individual parameters (psi space)
    map_eta : np.ndarray
        MAP estimates of random effects
    cond_mean_phi : np.ndarray
        Conditional mean of individual parameters (phi space)
    cond_mean_psi : np.ndarray
        Conditional mean of individual parameters (psi space)
    cond_mean_eta : np.ndarray
        Conditional mean of random effects
    cond_var_phi : np.ndarray
        Conditional variance of individual parameters
    cond_shrinkage : np.ndarray
        Shrinkage estimates for each parameter
    phi_samp : np.ndarray
        MCMC samples from conditional distribution
    fim : np.ndarray
        Fisher Information Matrix
    se_fixed : np.ndarray
        Standard errors of fixed effects
    conf_int : pd.DataFrame
        Confidence intervals for parameters
    parpop : np.ndarray
        Population parameters at each iteration
    allpar : np.ndarray
        All parameters at each iteration
    predictions : pd.DataFrame
        Predictions DataFrame
    ires : np.ndarray
        Individual residuals
    wres : np.ndarray
        Weighted residuals
    pd_ : np.ndarray
        Prediction discrepancies
    """

    def __init__(self):
        # Core estimation results
        self.fixed_effects = None
        self.omega = None
        self.respar = None

        # Likelihood and information criteria
        self.ll = None
        self.aic = None
        self.bic = None
        self.ll_is = None
        self.aic_is = None
        self.bic_is = None
        self.ll_gq = None
        self.aic_gq = None
        self.bic_gq = None
        self.npar_est = None

        # Individual parameter estimates
        self.mean_phi = None
        self.map_phi = None
        self.map_psi = None
        self.map_eta = None
        self.cond_mean_phi = None
        self.cond_mean_psi = None
        self.cond_mean_eta = None
        self.cond_var_phi = None

        # Conditional distribution estimation
        self.cond_shrinkage = None
        self.phi_samp = None

        # FIM and standard errors
        self.fim = None
        self.se_fixed = None

        # Confidence intervals
        self._conf_int = None

        # Iteration history
        self.parpop = None  # Population parameters at each iteration (n_iter, n_fixed)
        self.allpar = None  # All parameters at each iteration (n_iter, n_total)

        # Predictions and residuals
        self._predictions = None
        self.ires = None  # Individual residuals
        self.wres = None  # Weighted residuals
        self.pd_ = None  # Prediction discrepancies

    @property
    def conf_int(self) -> Optional[pd.DataFrame]:
        """Get confidence intervals DataFrame."""
        return self._conf_int

    @conf_int.setter
    def conf_int(self, value):
        """Set confidence intervals DataFrame."""
        self._conf_int = value

    @property
    def predictions(self) -> Optional[pd.DataFrame]:
        """Get predictions DataFrame."""
        return self._predictions

    @predictions.setter
    def predictions(self, value):
        """Set predictions DataFrame."""
        self._predictions = value

    def compute_confidence_intervals(
        self, alpha: float = 0.05, param_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute confidence intervals for all fixed effect parameters.

        Parameters
        ----------
        alpha : float
            Significance level (default: 0.05 for 95% CI)
        param_names : list of str, optional
            Parameter names for the DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: parameter, estimate, se, lower, upper, rse

        Notes
        -----
        Confidence intervals are computed as:
        - lower = estimate - z * se
        - upper = estimate + z * se
        where z = norm.ppf(1 - alpha/2)
        """
        # Get estimates from fixed_effects or mean_phi
        if self.fixed_effects is not None:
            estimates = np.atleast_1d(self.fixed_effects)
        elif self.mean_phi is not None:
            # Use the first row of mean_phi as population estimates
            estimates = np.atleast_1d(
                self.mean_phi[0] if self.mean_phi.ndim > 1 else self.mean_phi
            )
        else:
            raise ValueError("Fixed effects not available. Run SAEM algorithm first.")

        n_params = len(estimates)

        # Get standard errors
        if self.se_fixed is not None:
            se = np.atleast_1d(self.se_fixed)
        elif self.fim is not None:
            # Compute SE from FIM
            try:
                cov = np.linalg.inv(self.fim)
                se = np.sqrt(np.diag(cov)[:n_params])
            except np.linalg.LinAlgError:
                se = np.full(n_params, np.nan)
        else:
            se = np.full(n_params, np.nan)

        # Ensure se has correct length
        if len(se) < n_params:
            se = np.concatenate([se, np.full(n_params - len(se), np.nan)])
        se = se[:n_params]

        # Compute confidence bounds
        z = stats.norm.ppf(1 - alpha / 2)
        lower = estimates - z * se
        upper = estimates + z * se

        # Compute relative standard error (%)
        rse = np.where(estimates != 0, 100 * se / np.abs(estimates), np.nan)

        # Generate parameter names if not provided
        if param_names is None:
            param_names = [f"theta{i+1}" for i in range(n_params)]
        elif len(param_names) < n_params:
            param_names = list(param_names) + [
                f"theta{i+1}" for i in range(len(param_names), n_params)
            ]

        # Create DataFrame
        self._conf_int = pd.DataFrame(
            {
                "parameter": param_names[:n_params],
                "estimate": estimates,
                "se": se,
                "lower": lower,
                "upper": upper,
                "rse": rse,
            }
        )

        return self._conf_int

    def compute_residuals(self, saemix_object: "SaemixObject") -> None:
        """
        Compute various residuals for model diagnostics.

        Parameters
        ----------
        saemix_object : SaemixObject
            The fitted SAEM object

        Notes
        -----
        Computes:
        - ires: Individual residuals (yobs - ipred)
        - wres: Weighted residuals (ires / g)
        - pd_: Prediction discrepancies
        """
        from saemix.algorithm.map_estimation import error_function
        from saemix.algorithm.predict import saemix_predict
        from saemix.utils import cutoff, transphi

        data = saemix_object.data
        model = saemix_object.model

        # Get observations
        yobs = data.data[data.name_response].values
        xind = data.data[data.name_predictors].values
        index = data.data["index"].values

        # Get predictions
        pred_dict = saemix_predict(saemix_object, type=["ipred", "ppred"])
        ipred = pred_dict.get("ipred", np.zeros_like(yobs))
        ppred = pred_dict.get("ppred", np.zeros_like(yobs))

        # Compute individual residuals
        self.ires = yobs - ipred

        # Compute weighted residuals
        if model.modeltype == "structural" and self.respar is not None:
            ytype = data.data["ytype"].values if "ytype" in data.data.columns else None
            g = error_function(ipred, self.respar, model.error_model, ytype)
            self.wres = self.ires / cutoff(g)
        else:
            self.wres = self.ires

        # Compute prediction discrepancies (population-based)
        self.pd_ = yobs - ppred

    def build_predictions_dataframe(
        self, saemix_object: "SaemixObject"
    ) -> pd.DataFrame:
        """
        Build a comprehensive predictions DataFrame.

        Parameters
        ----------
        saemix_object : SaemixObject
            The fitted SAEM object

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: id, time, yobs, ppred, ipred, ires, wres
        """
        from saemix.algorithm.predict import saemix_predict

        data = saemix_object.data

        # Get observations
        yobs = data.data[data.name_response].values
        time = data.data[data.name_predictors[0]].values
        id_col = data.data[data.name_group].values

        # Get predictions
        pred_dict = saemix_predict(saemix_object, type=["ipred", "ppred"])
        ipred = pred_dict.get("ipred", np.zeros_like(yobs))
        ppred = pred_dict.get("ppred", np.zeros_like(yobs))

        # Compute residuals if not already done
        if self.ires is None:
            self.compute_residuals(saemix_object)

        # Build DataFrame
        self._predictions = pd.DataFrame(
            {
                "id": id_col,
                "time": time,
                "yobs": yobs,
                "ppred": ppred,
                "ipred": ipred,
                "ires": self.ires,
                "wres": self.wres,
            }
        )

        return self._predictions


class SaemixObject:
    """
    Main object containing data, model, options, and results.

    Parameters
    ----------
    data : SaemixData
        The data object
    model : SaemixModel
        The model object
    options : dict
        Algorithm options
    """

    def __init__(self, data: "SaemixData", model: "SaemixModel", options: dict):
        self.data = data
        self.model = model
        self.options = options
        self.results = SaemixRes()

    def psi(self, type: str = "mode") -> np.ndarray:
        """
        Get individual parameter estimates in psi space.

        Parameters
        ----------
        type : str
            "mode" for MAP estimates, "mean" for conditional mean

        Returns
        -------
        np.ndarray
            Individual parameter estimates
        """
        if type == "mode":
            return self.results.map_psi
        else:
            return self.results.cond_mean_psi

    def phi(self, type: str = "mode") -> np.ndarray:
        """
        Get individual parameter estimates in phi space.

        Parameters
        ----------
        type : str
            "mode" for MAP estimates, "mean" for conditional mean

        Returns
        -------
        np.ndarray
            Individual parameter estimates
        """
        if type == "mode":
            return self.results.map_phi
        else:
            return self.results.cond_mean_phi

    def eta(self, type: str = "mode") -> np.ndarray:
        """
        Get random effect estimates.

        Parameters
        ----------
        type : str
            "mode" for MAP estimates, "mean" for conditional mean

        Returns
        -------
        np.ndarray
            Random effect estimates
        """
        if type == "mode":
            return self.results.map_eta
        else:
            return self.results.cond_mean_eta

    def predict(self, type: str = "ipred", newdata: Optional = None) -> np.ndarray:
        """
        Get predictions.

        Parameters
        ----------
        type : str
            Prediction type: "ipred" (MAP individual), "ppred" (population),
            "ypred" (population mean), "icpred" (conditional mean individual)
        newdata : Optional
            New data for prediction (not yet implemented)

        Returns
        -------
        np.ndarray
            Predictions
        """
        from saemix.algorithm.predict import saemix_predict

        if newdata is not None:
            raise NotImplementedError("newdata prediction not yet implemented")
        pred_dict = saemix_predict(self, type=[type])
        return pred_dict.get(type, None)

    def summary(self):
        """Print a summary of the fitted model."""
        print("SaemixObject summary")
        print(f"  Model: {self.model.description}")
        print(f"  Subjects: {self.data.n_subjects}")
        print(f"  Observations: {self.data.n_total_obs}")

        if self.results.fixed_effects is not None:
            print(f"  Fixed effects estimated: {len(self.results.fixed_effects)}")
            print(f"  Fixed effects: {self.results.fixed_effects}")

        if self.results.ll is not None:
            print(f"  Log-likelihood: {self.results.ll:.4f}")

        if self.results.aic is not None:
            print(f"  AIC: {self.results.aic:.4f}")

        if self.results.bic is not None:
            print(f"  BIC: {self.results.bic:.4f}")

    def __repr__(self):
        return f"SaemixObject:\n  Data: {self.data.n_subjects} subjects\n  Model: {self.model.description}"
