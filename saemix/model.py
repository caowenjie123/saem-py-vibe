import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np


class SaemixModel:
    def __init__(
        self,
        model: Callable,
        psi0: Union[np.ndarray, List[List[float]], Dict[str, float]],
        description: str = "",
        modeltype: str = "structural",
        error_model: Union[str, List[str]] = "constant",
        transform_par: Optional[List[int]] = None,
        fixed_estim: Optional[List[int]] = None,
        covariate_model: Optional[np.ndarray] = None,
        covariance_model: Optional[np.ndarray] = None,
        omega_init: Optional[np.ndarray] = None,
        error_init: Optional[List[float]] = None,
        name_modpar: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.description = description
        self.modeltype = modeltype
        self.verbose = verbose

        self._validate_model()
        self.psi0 = self._process_psi0(psi0, name_modpar)
        self.n_parameters = self.psi0.shape[1]

        if transform_par is None:
            transform_par = [0] * self.n_parameters
        self.transform_par = np.array(transform_par)

        if fixed_estim is None:
            fixed_estim = [1] * self.n_parameters
        self.fixed_estim = np.array(fixed_estim)

        if error_model is None or error_model == "":
            error_model = "constant"
        if isinstance(error_model, str):
            error_model = [error_model]
        self.error_model = error_model

        if covariate_model is None:
            covariate_model = np.zeros((0, self.n_parameters))
        else:
            covariate_model = np.array(covariate_model)
            if covariate_model.ndim == 1:
                covariate_model = covariate_model.reshape(1, -1)
        self.covariate_model = covariate_model

        if covariance_model is None:
            covariance_model = np.eye(self.n_parameters)
        else:
            covariance_model = np.array(covariance_model)
        self.covariance_model = covariance_model

        self.indx_omega = np.where(np.diag(covariance_model) > 0)[0]

        if omega_init is None:
            omega_init = np.eye(self.n_parameters)
            diag_omega = np.ones(self.n_parameters)
            j1 = np.where(self.transform_par == 0)[0]
            if len(j1) > 0:
                for i in j1:
                    val = self.psi0[0, i]
                    diag_omega[i] = max(val**2, 1)
            omega_init = np.diag(diag_omega)
        else:
            omega_init = np.array(omega_init)
        self.omega_init = omega_init

        if error_init is None:
            error_init = []
            for em in self.error_model:
                if em == "constant" or em == "exponential":
                    error_init.extend([1, 0])
                elif em == "proportional":
                    error_init.extend([0, 1])
                elif em == "combined":
                    error_init.extend([1, 1])
        self.error_init = np.array(error_init)

        self._validate_sizes()
        self._compute_betaest_model()
        self._compute_indx_res()

    def _validate_model(self):
        if not callable(self.model):
            raise TypeError("model must be a callable function")

        sig = inspect.signature(self.model)
        params = list(sig.parameters.keys())
        if len(params) != 3:
            raise ValueError(
                "model function must accept exactly 3 arguments: psi, id, xidep"
            )

    def _process_psi0(self, psi0, name_modpar):
        if isinstance(psi0, dict):
            if name_modpar is None:
                name_modpar = list(psi0.keys())
            psi0_arr = np.array([[psi0[k] for k in name_modpar]])
        elif isinstance(psi0, list):
            psi0_arr = np.array(psi0)
            if psi0_arr.ndim == 1:
                psi0_arr = psi0_arr.reshape(1, -1)
        else:
            psi0_arr = np.array(psi0)
            if psi0_arr.ndim == 1:
                psi0_arr = psi0_arr.reshape(1, -1)

        if name_modpar is None:
            if hasattr(psi0_arr, "columns") or (
                hasattr(psi0, "columns") if not isinstance(psi0, np.ndarray) else False
            ):
                name_modpar = (
                    list(psi0_arr.columns)
                    if hasattr(psi0_arr, "columns")
                    else list(psi0.columns)
                )
            else:
                name_modpar = [f"theta{i+1}" for i in range(psi0_arr.shape[1])]
        self.name_modpar = name_modpar

        return psi0_arr

    def _validate_sizes(self):
        npar = self.n_parameters

        if len(self.transform_par) != npar:
            raise ValueError(
                f"transform_par length ({len(self.transform_par)}) must match number of parameters ({npar})"
            )

        if len(self.fixed_estim) != npar:
            raise ValueError(
                f"fixed_estim length ({len(self.fixed_estim)}) must match number of parameters ({npar})"
            )

        if self.covariate_model.shape[1] != npar:
            raise ValueError(
                f"covariate_model columns ({self.covariate_model.shape[1]}) must match number of parameters ({npar})"
            )

        if self.covariance_model.shape[0] != self.covariance_model.shape[1]:
            raise ValueError("covariance_model must be square")

        if self.covariance_model.shape[0] != npar:
            raise ValueError(
                f"covariance_model size ({self.covariance_model.shape[0]}) must match number of parameters ({npar})"
            )

        if self.omega_init.shape[0] != self.omega_init.shape[1]:
            raise ValueError("omega_init must be square")

        if self.omega_init.shape[0] != npar:
            raise ValueError(
                f"omega_init size ({self.omega_init.shape[0]}) must match number of parameters ({npar})"
            )

        if np.sum(self.fixed_estim * np.diag(self.covariance_model)) == 0:
            raise ValueError("At least one parameter with IIV must be estimated")

        valid_models = ["constant", "proportional", "combined", "exponential"]
        for em in self.error_model:
            if em not in valid_models:
                raise ValueError(
                    f"Invalid error model: {em}. Must be one of {valid_models}"
                )

        if self.modeltype not in ["structural", "likelihood"]:
            raise ValueError(
                f"Invalid modeltype: {self.modeltype}. Must be 'structural' or 'likelihood'"
            )

    def _compute_betaest_model(self):
        npar = self.n_parameters
        ncov = self.covariate_model.shape[0]
        betaest = np.ones((1 + ncov, npar))
        if ncov > 0:
            betaest[1:, :] = self.covariate_model
        self.betaest_model = betaest

    def _compute_indx_res(self):
        indx_res = []
        for i, em in enumerate(self.error_model):
            if em == "constant" or em == "exponential":
                indx_res.append(1 + 2 * i)
            elif em == "proportional":
                indx_res.append(2 + 2 * i)
            elif em == "combined":
                indx_res.extend([1 + 2 * i, 2 + 2 * i])
        self.indx_res = np.array(indx_res) - 1

        error_init_mask = np.ones(len(self.error_init), dtype=bool)
        for i in range(len(self.error_init)):
            if i not in self.indx_res:
                error_init_mask[i] = False
        self.error_init[~error_init_mask] = 0

    def __repr__(self):
        return f"SaemixModel: {self.description}\n  Parameters: {self.n_parameters}\n  Model type: {self.modeltype}\n  Error model: {self.error_model}"


def saemix_model(
    model: Callable,
    psi0: Union[np.ndarray, List[List[float]], Dict[str, float]],
    description: str = "",
    modeltype: str = "structural",
    error_model: Union[str, List[str]] = "constant",
    transform_par: Optional[List[int]] = None,
    fixed_estim: Optional[List[int]] = None,
    covariate_model: Optional[np.ndarray] = None,
    covariance_model: Optional[np.ndarray] = None,
    omega_init: Optional[np.ndarray] = None,
    error_init: Optional[List[float]] = None,
    name_modpar: Optional[List[str]] = None,
    verbose: bool = True,
) -> SaemixModel:
    return SaemixModel(
        model=model,
        psi0=psi0,
        description=description,
        modeltype=modeltype,
        error_model=error_model,
        transform_par=transform_par,
        fixed_estim=fixed_estim,
        covariate_model=covariate_model,
        covariance_model=covariance_model,
        omega_init=omega_init,
        error_init=error_init,
        name_modpar=name_modpar,
        verbose=verbose,
    )
