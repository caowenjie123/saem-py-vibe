"""
Stepwise Regression Module

This module provides functions for stepwise covariate selection in SAEM models
using information criteria (BIC, AIC).

Feature: saemix-python-enhancement
Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
"""

from typing import Dict, List, Optional

import numpy as np

from saemix.compare import aic, bic
from saemix.model import saemix_model
from saemix.results import SaemixObject


def forward_procedure(
    saemix_object: SaemixObject,
    covariates: Optional[List[str]] = None,
    parameters: Optional[List[str]] = None,
    trace: bool = True,
    criterion: str = "BIC",
) -> SaemixObject:
    """
    Forward selection of covariates.

    Starting from a model with no covariates, iteratively adds the covariate
    that most improves the information criterion until no improvement is found.

    Parameters
    ----------
    saemix_object : SaemixObject
        Fitted SAEM result object
    covariates : list of str, optional
        List of covariate names to consider. If None, uses all covariates
        from the data.
    parameters : list of str, optional
        List of parameter names to consider for covariate effects.
        If None, uses all parameters.
    trace : bool
        If True, prints the selection process including criterion values
        at each step.
    criterion : str
        Information criterion to use: 'BIC' (default) or 'AIC'

    Returns
    -------
    SaemixObject
        Fitted model with the optimal covariate model

    Raises
    ------
    ValueError
        If no covariates are available for selection
        If criterion is not 'BIC' or 'AIC'

    Notes
    -----
    The forward procedure adds covariates one at a time, selecting the
    covariate-parameter combination that provides the largest improvement
    in the information criterion.

    Examples
    --------
    >>> result = saemix(model, data, control)
    >>> result_forward = forward_procedure(result, trace=True)
    """
    _validate_criterion(criterion)

    # Get available covariates
    available_covariates = _get_available_covariates(saemix_object, covariates)
    if len(available_covariates) == 0:
        raise ValueError("No covariates available for selection")

    # Get available parameters
    available_parameters = _get_available_parameters(saemix_object, parameters)

    # Initialize with empty covariate model
    current_model = _create_empty_covariate_model(saemix_object)
    current_object = _fit_model_with_covariates(saemix_object, current_model)
    current_criterion = _get_criterion_value(current_object, criterion)

    if trace:
        print(f"Forward Selection using {criterion}")
        print(f"Initial {criterion}: {current_criterion:.4f}")
        print("-" * 50)

    # Track which covariate-parameter combinations are in the model
    included = set()

    improved = True
    step = 0

    while improved:
        improved = False
        best_criterion = current_criterion
        best_combination = None
        best_object = None

        # Try adding each covariate-parameter combination
        for cov in available_covariates:
            for param_idx, param in enumerate(available_parameters):
                combination = (cov, param_idx)
                if combination in included:
                    continue

                # Create new covariate model with this combination
                test_model = _add_covariate_to_model(
                    current_model, cov, param_idx, saemix_object
                )

                try:
                    test_object = _fit_model_with_covariates(saemix_object, test_model)
                    test_criterion = _get_criterion_value(test_object, criterion)

                    if test_criterion < best_criterion:
                        best_criterion = test_criterion
                        best_combination = combination
                        best_object = test_object
                except Exception:
                    # Skip combinations that fail to fit
                    continue

        if best_combination is not None:
            improved = True
            step += 1
            included.add(best_combination)
            current_model = _add_covariate_to_model(
                current_model, best_combination[0], best_combination[1], saemix_object
            )
            current_object = best_object
            current_criterion = best_criterion

            if trace:
                cov_name = best_combination[0]
                param_name = available_parameters[best_combination[1]]
                print(f"Step {step}: Added {cov_name} on {param_name}")
                print(f"  {criterion}: {current_criterion:.4f}")

    if trace:
        print("-" * 50)
        print(f"Final {criterion}: {current_criterion:.4f}")
        print(f"Covariates included: {len(included)}")

    return current_object


def backward_procedure(
    saemix_object: SaemixObject,
    trace: bool = True,
    criterion: str = "BIC",
) -> SaemixObject:
    """
    Backward elimination of covariates.

    Starting from the current model, iteratively removes the covariate
    that least worsens (or most improves) the information criterion
    until no more can be removed.

    Parameters
    ----------
    saemix_object : SaemixObject
        Fitted SAEM result object with covariates
    trace : bool
        If True, prints the selection process including criterion values
        at each step.
    criterion : str
        Information criterion to use: 'BIC' (default) or 'AIC'

    Returns
    -------
    SaemixObject
        Fitted model with the optimal covariate model

    Raises
    ------
    ValueError
        If criterion is not 'BIC' or 'AIC'

    Notes
    -----
    The backward procedure removes covariates one at a time, selecting the
    covariate-parameter combination whose removal provides the largest
    improvement (or smallest worsening) in the information criterion.

    Examples
    --------
    >>> result = saemix(model, data, control)
    >>> result_backward = backward_procedure(result, trace=True)
    """
    _validate_criterion(criterion)

    # Get current covariate model
    current_model = _get_current_covariate_model(saemix_object)
    current_object = saemix_object
    current_criterion = _get_criterion_value(current_object, criterion)

    if trace:
        print(f"Backward Elimination using {criterion}")
        print(f"Initial {criterion}: {current_criterion:.4f}")
        print("-" * 50)

    # Get list of included covariate-parameter combinations
    included = _get_included_combinations(current_model, saemix_object)

    if len(included) == 0:
        if trace:
            print("No covariates to remove")
        return saemix_object

    improved = True
    step = 0

    while improved and len(included) > 0:
        improved = False
        best_criterion = current_criterion
        best_combination = None
        best_object = None

        # Try removing each covariate-parameter combination
        for combination in list(included):
            # Create new covariate model without this combination
            test_model = _remove_covariate_from_model(
                current_model, combination[0], combination[1], saemix_object
            )

            try:
                test_object = _fit_model_with_covariates(saemix_object, test_model)
                test_criterion = _get_criterion_value(test_object, criterion)

                if test_criterion < best_criterion:
                    best_criterion = test_criterion
                    best_combination = combination
                    best_object = test_object
            except Exception:
                # Skip combinations that fail to fit
                continue

        if best_combination is not None:
            improved = True
            step += 1
            included.remove(best_combination)
            current_model = _remove_covariate_from_model(
                current_model, best_combination[0], best_combination[1], saemix_object
            )
            current_object = best_object
            current_criterion = best_criterion

            if trace:
                param_names = saemix_object.model.name_modpar
                cov_name = best_combination[0]
                param_name = param_names[best_combination[1]]
                print(f"Step {step}: Removed {cov_name} from {param_name}")
                print(f"  {criterion}: {current_criterion:.4f}")

    if trace:
        print("-" * 50)
        print(f"Final {criterion}: {current_criterion:.4f}")
        print(f"Covariates remaining: {len(included)}")

    return current_object


def stepwise_procedure(
    saemix_object: SaemixObject,
    direction: str = "both",
    covariate_init: Optional[Dict] = None,
    trace: bool = True,
    criterion: str = "BIC",
) -> SaemixObject:
    """
    Bidirectional stepwise covariate selection.

    Alternates between forward selection and backward elimination steps
    until no improvement is found in either direction.

    Parameters
    ----------
    saemix_object : SaemixObject
        Fitted SAEM result object
    direction : str
        Selection direction: 'both' (default), 'forward', or 'backward'
    covariate_init : dict, optional
        Initial covariate model specification. Keys are covariate names,
        values are lists of parameter indices.
    trace : bool
        If True, prints the selection process including criterion values
        at each step.
    criterion : str
        Information criterion to use: 'BIC' (default) or 'AIC'

    Returns
    -------
    SaemixObject
        Fitted model with the optimal covariate model

    Raises
    ------
    ValueError
        If direction is not 'both', 'forward', or 'backward'
        If criterion is not 'BIC' or 'AIC'

    Notes
    -----
    The stepwise procedure combines forward and backward steps:
    1. Try to add a covariate (forward step)
    2. Try to remove a covariate (backward step)
    3. Repeat until no improvement in either direction

    Examples
    --------
    >>> result = saemix(model, data, control)
    >>> result_stepwise = stepwise_procedure(result, direction='both', trace=True)
    """
    _validate_criterion(criterion)
    _validate_direction(direction)

    if direction == "forward":
        return forward_procedure(saemix_object, trace=trace, criterion=criterion)
    elif direction == "backward":
        return backward_procedure(saemix_object, trace=trace, criterion=criterion)

    # direction == 'both'
    # Get available covariates
    available_covariates = _get_available_covariates(saemix_object, None)
    available_parameters = _get_available_parameters(saemix_object, None)

    # Initialize covariate model
    if covariate_init is not None:
        current_model = _create_covariate_model_from_init(covariate_init, saemix_object)
    else:
        current_model = _create_empty_covariate_model(saemix_object)

    current_object = _fit_model_with_covariates(saemix_object, current_model)
    current_criterion = _get_criterion_value(current_object, criterion)

    if trace:
        print(f"Stepwise Selection (both directions) using {criterion}")
        print(f"Initial {criterion}: {current_criterion:.4f}")
        print("-" * 50)

    # Track included combinations
    included = _get_included_combinations(current_model, saemix_object)

    improved = True
    step = 0

    while improved:
        improved = False

        # Forward step: try adding
        best_add_criterion = current_criterion
        best_add_combination = None
        best_add_object = None

        for cov in available_covariates:
            for param_idx, param in enumerate(available_parameters):
                combination = (cov, param_idx)
                if combination in included:
                    continue

                test_model = _add_covariate_to_model(
                    current_model, cov, param_idx, saemix_object
                )

                try:
                    test_object = _fit_model_with_covariates(saemix_object, test_model)
                    test_criterion = _get_criterion_value(test_object, criterion)

                    if test_criterion < best_add_criterion:
                        best_add_criterion = test_criterion
                        best_add_combination = combination
                        best_add_object = test_object
                except Exception:
                    continue

        # Backward step: try removing
        best_remove_criterion = current_criterion
        best_remove_combination = None
        best_remove_object = None

        for combination in list(included):
            test_model = _remove_covariate_from_model(
                current_model, combination[0], combination[1], saemix_object
            )

            try:
                test_object = _fit_model_with_covariates(saemix_object, test_model)
                test_criterion = _get_criterion_value(test_object, criterion)

                if test_criterion < best_remove_criterion:
                    best_remove_criterion = test_criterion
                    best_remove_combination = combination
                    best_remove_object = test_object
            except Exception:
                continue

        # Decide which action to take
        if best_add_combination is not None and best_add_criterion < current_criterion:
            if (
                best_remove_combination is None
                or best_add_criterion <= best_remove_criterion
            ):
                # Add is better
                improved = True
                step += 1
                included.add(best_add_combination)
                current_model = _add_covariate_to_model(
                    current_model,
                    best_add_combination[0],
                    best_add_combination[1],
                    saemix_object,
                )
                current_object = best_add_object
                current_criterion = best_add_criterion

                if trace:
                    cov_name = best_add_combination[0]
                    param_name = available_parameters[best_add_combination[1]]
                    print(f"Step {step}: Added {cov_name} on {param_name}")
                    print(f"  {criterion}: {current_criterion:.4f}")

        if (
            not improved
            and best_remove_combination is not None
            and best_remove_criterion < current_criterion
        ):
            # Remove is better
            improved = True
            step += 1
            included.remove(best_remove_combination)
            current_model = _remove_covariate_from_model(
                current_model,
                best_remove_combination[0],
                best_remove_combination[1],
                saemix_object,
            )
            current_object = best_remove_object
            current_criterion = best_remove_criterion

            if trace:
                param_names = saemix_object.model.name_modpar
                cov_name = best_remove_combination[0]
                param_name = param_names[best_remove_combination[1]]
                print(f"Step {step}: Removed {cov_name} from {param_name}")
                print(f"  {criterion}: {current_criterion:.4f}")

    if trace:
        print("-" * 50)
        print(f"Final {criterion}: {current_criterion:.4f}")
        print(f"Covariates in final model: {len(included)}")

    return current_object


# Helper functions


def _validate_criterion(criterion: str) -> None:
    """Validate the criterion parameter."""
    valid_criteria = ["BIC", "AIC"]
    if criterion not in valid_criteria:
        raise ValueError(
            f"criterion must be one of {valid_criteria}, got '{criterion}'"
        )


def _validate_direction(direction: str) -> None:
    """Validate the direction parameter."""
    valid_directions = ["forward", "backward", "both"]
    if direction not in valid_directions:
        raise ValueError(
            f"direction must be one of {valid_directions}, got '{direction}'"
        )


def _get_available_covariates(
    saemix_object: SaemixObject, covariates: Optional[List[str]]
) -> List[str]:
    """Get list of available covariates."""
    data = saemix_object.data

    if covariates is not None:
        # Validate provided covariates exist in data
        available = []
        for cov in covariates:
            if cov in data.name_covariates:
                available.append(cov)
        return available

    return list(data.name_covariates) if data.name_covariates else []


def _get_available_parameters(
    saemix_object: SaemixObject, parameters: Optional[List[str]]
) -> List[str]:
    """Get list of available parameters."""
    model = saemix_object.model

    if parameters is not None:
        return parameters

    return model.name_modpar


def _create_empty_covariate_model(saemix_object: SaemixObject) -> np.ndarray:
    """Create an empty covariate model matrix."""
    n_covariates = (
        len(saemix_object.data.name_covariates)
        if saemix_object.data.name_covariates
        else 0
    )
    n_parameters = saemix_object.model.n_parameters

    if n_covariates == 0:
        return np.zeros((0, n_parameters))

    return np.zeros((n_covariates, n_parameters))


def _get_current_covariate_model(saemix_object: SaemixObject) -> np.ndarray:
    """Get the current covariate model matrix."""
    return saemix_object.model.covariate_model.copy()


def _get_included_combinations(
    covariate_model: np.ndarray, saemix_object: SaemixObject
) -> set:
    """Get set of included covariate-parameter combinations."""
    included = set()

    if covariate_model.size == 0:
        return included

    covariates = saemix_object.data.name_covariates
    if not covariates:
        return included

    for cov_idx, cov in enumerate(covariates):
        if cov_idx >= covariate_model.shape[0]:
            continue
        for param_idx in range(covariate_model.shape[1]):
            if covariate_model[cov_idx, param_idx] != 0:
                included.add((cov, param_idx))

    return included


def _add_covariate_to_model(
    current_model: np.ndarray,
    covariate: str,
    param_idx: int,
    saemix_object: SaemixObject,
) -> np.ndarray:
    """Add a covariate-parameter combination to the model."""
    covariates = saemix_object.data.name_covariates
    n_parameters = saemix_object.model.n_parameters

    if not covariates:
        return current_model.copy()

    cov_idx = covariates.index(covariate) if covariate in covariates else -1
    if cov_idx < 0:
        return current_model.copy()

    # Ensure model has correct shape
    if current_model.size == 0:
        new_model = np.zeros((len(covariates), n_parameters))
    else:
        new_model = current_model.copy()
        if new_model.shape[0] < len(covariates):
            # Expand model
            expanded = np.zeros((len(covariates), n_parameters))
            expanded[: new_model.shape[0], :] = new_model
            new_model = expanded

    new_model[cov_idx, param_idx] = 1
    return new_model


def _remove_covariate_from_model(
    current_model: np.ndarray,
    covariate: str,
    param_idx: int,
    saemix_object: SaemixObject,
) -> np.ndarray:
    """Remove a covariate-parameter combination from the model."""
    covariates = saemix_object.data.name_covariates

    if not covariates or current_model.size == 0:
        return current_model.copy()

    cov_idx = covariates.index(covariate) if covariate in covariates else -1
    if cov_idx < 0 or cov_idx >= current_model.shape[0]:
        return current_model.copy()

    new_model = current_model.copy()
    new_model[cov_idx, param_idx] = 0
    return new_model


def _create_covariate_model_from_init(
    covariate_init: Dict, saemix_object: SaemixObject
) -> np.ndarray:
    """Create covariate model from initialization dictionary."""
    covariates = saemix_object.data.name_covariates
    n_parameters = saemix_object.model.n_parameters

    if not covariates:
        return np.zeros((0, n_parameters))

    model = np.zeros((len(covariates), n_parameters))

    for cov, param_indices in covariate_init.items():
        if cov in covariates:
            cov_idx = covariates.index(cov)
            for param_idx in param_indices:
                if 0 <= param_idx < n_parameters:
                    model[cov_idx, param_idx] = 1

    return model


def _fit_model_with_covariates(
    saemix_object: SaemixObject, covariate_model: np.ndarray
) -> SaemixObject:
    """Fit a model with the specified covariate model."""
    from saemix.main import saemix

    original_model = saemix_object.model
    data = saemix_object.data
    options = saemix_object.options.copy()

    # Suppress output during stepwise
    options["display_progress"] = False
    options["warnings"] = False

    # Create new model with updated covariate model
    new_model = saemix_model(
        model=original_model.model,
        psi0=original_model.psi0,
        description=original_model.description,
        modeltype=original_model.modeltype,
        error_model=original_model.error_model,
        transform_par=list(original_model.transform_par),
        fixed_estim=list(original_model.fixed_estim),
        covariate_model=covariate_model if covariate_model.size > 0 else None,
        covariance_model=original_model.covariance_model,
        omega_init=original_model.omega_init,
        error_init=list(original_model.error_init),
        name_modpar=original_model.name_modpar,
        verbose=False,
    )

    # Fit the model
    result = saemix(model=new_model, data=data, control=options)

    return result


def _get_criterion_value(saemix_object: SaemixObject, criterion: str) -> float:
    """Get the information criterion value for a fitted model."""
    if criterion == "BIC":
        return bic(saemix_object, method="is")
    else:  # AIC
        return aic(saemix_object, method="is")
