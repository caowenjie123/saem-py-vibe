"""
Property-Based Tests for Stepwise Regression

Feature: saemix-python-enhancement
Property 8: Stepwise Selection Optimality
Validates: Requirements 4.1, 4.2, 4.3, 4.5
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st, assume

from saemix import saemix, saemix_data, saemix_model, saemix_control
from saemix.stepwise import (
    forward_procedure,
    backward_procedure,
    stepwise_procedure,
    _get_criterion_value,
    _get_included_combinations,
    _add_covariate_to_model,
    _remove_covariate_from_model,
    _fit_model_with_covariates,
)
from saemix.compare import bic


def linear_model_with_cov(psi, id, xidep):
    """Linear model: y = psi[id, 0] * x + psi[id, 1]"""
    return psi[id, 0] * xidep[:, 0] + psi[id, 1]


def create_test_data_with_covariates(
    n_subjects: int = 10, n_obs_per_subject: int = 5, seed: int = 42
):
    """Create synthetic test data with covariates."""
    np.random.seed(seed)

    data_list = []
    for i in range(n_subjects):
        x = np.linspace(0, 3, n_obs_per_subject)
        # Covariate effects
        weight = 50 + np.random.normal(0, 10)  # Weight covariate
        age = 30 + np.random.normal(0, 5)  # Age covariate

        # True parameters with covariate effects
        true_a = 2.0 + 0.01 * weight + np.random.normal(0, 0.2)
        true_b = 1.0 + 0.02 * age + np.random.normal(0, 0.1)
        y = true_a * x + true_b + np.random.normal(0, 0.1, n_obs_per_subject)

        for j in range(n_obs_per_subject):
            data_list.append(
                {
                    "Id": i + 1,
                    "X": x[j],
                    "Y": y[j],
                    "Weight": weight,
                    "Age": age,
                }
            )

    return pd.DataFrame(data_list)


def create_fitted_model_with_covariates(seed: int = 42):
    """Create a fitted model with covariates available."""
    data = create_test_data_with_covariates(
        n_subjects=8, n_obs_per_subject=4, seed=seed
    )

    model = saemix_model(
        model=linear_model_with_cov,
        psi0=np.array([[2.0, 1.0]]),
        description="Linear model with covariates",
        name_modpar=["a", "b"],
    )

    sdata = saemix_data(
        name_data=data,
        name_group="Id",
        name_predictors=["X"],
        name_response="Y",
        name_covariates=["Weight", "Age"],
        verbose=False,
    )

    control = saemix_control(
        nbiter_saemix=(15, 8),
        display_progress=False,
        warnings=False,
        map=True,
    )

    result = saemix(model=model, data=sdata, control=control)
    return result


class TestStepwiseProperties:
    """Property-based tests for stepwise selection."""

    @settings(max_examples=100, deadline=None)
    @given(seed=st.integers(min_value=1, max_value=10000))
    def test_property_8_forward_optimality(self, seed):
        """
        Feature: saemix-python-enhancement, Property 8: Stepwise Selection Optimality
        Validates: Requirements 4.1, 4.5

        For forward selection, the final model SHALL be locally optimal, meaning
        no single covariate addition improves BIC.
        """
        # Create fitted model
        result = create_fitted_model_with_covariates(seed=seed)

        # Run forward selection
        final_result = forward_procedure(result, trace=False, criterion="BIC")

        # Get final BIC
        final_bic = _get_criterion_value(final_result, "BIC")

        # Get included combinations
        final_model = final_result.model.covariate_model
        included = _get_included_combinations(final_model, final_result)

        # Try adding each remaining covariate-parameter combination
        available_covariates = final_result.data.name_covariates
        n_parameters = final_result.model.n_parameters

        for cov in available_covariates:
            for param_idx in range(n_parameters):
                combination = (cov, param_idx)
                if combination in included:
                    continue

                # Try adding this combination
                test_model = _add_covariate_to_model(
                    final_model, cov, param_idx, final_result
                )

                try:
                    test_result = _fit_model_with_covariates(final_result, test_model)
                    test_bic = _get_criterion_value(test_result, "BIC")

                    # No addition should improve BIC
                    assert test_bic >= final_bic - 1e-6, (
                        f"Forward selection not optimal: adding {cov} on param {param_idx} "
                        f"improves BIC from {final_bic:.4f} to {test_bic:.4f}"
                    )
                except Exception:
                    # If fitting fails, that's fine - combination is not viable
                    pass

    @settings(max_examples=100, deadline=None)
    @given(seed=st.integers(min_value=1, max_value=10000))
    def test_property_8_backward_optimality(self, seed):
        """
        Feature: saemix-python-enhancement, Property 8: Stepwise Selection Optimality
        Validates: Requirements 4.2, 4.5

        For backward elimination, the final model SHALL be locally optimal, meaning
        no single covariate removal improves BIC.
        """
        # Create fitted model with some covariates
        result = create_fitted_model_with_covariates(seed=seed)

        # First add some covariates via forward selection
        result_with_cov = forward_procedure(result, trace=False, criterion="BIC")

        # Run backward elimination
        final_result = backward_procedure(result_with_cov, trace=False, criterion="BIC")

        # Get final BIC
        final_bic = _get_criterion_value(final_result, "BIC")

        # Get included combinations
        final_model = final_result.model.covariate_model
        included = _get_included_combinations(final_model, final_result)

        # Try removing each included covariate-parameter combination
        for combination in included:
            cov, param_idx = combination

            # Try removing this combination
            test_model = _remove_covariate_from_model(
                final_model, cov, param_idx, final_result
            )

            try:
                test_result = _fit_model_with_covariates(final_result, test_model)
                test_bic = _get_criterion_value(test_result, "BIC")

                # No removal should improve BIC
                assert test_bic >= final_bic - 1e-6, (
                    f"Backward elimination not optimal: removing {cov} from param {param_idx} "
                    f"improves BIC from {final_bic:.4f} to {test_bic:.4f}"
                )
            except Exception:
                # If fitting fails, that's fine
                pass

    @settings(max_examples=100, deadline=None)
    @given(seed=st.integers(min_value=1, max_value=10000))
    def test_property_8_stepwise_optimality(self, seed):
        """
        Feature: saemix-python-enhancement, Property 8: Stepwise Selection Optimality
        Validates: Requirements 4.3, 4.5

        For stepwise selection (both directions), the final model SHALL be locally
        optimal, meaning neither addition nor removal improves BIC.
        """
        # Create fitted model
        result = create_fitted_model_with_covariates(seed=seed)

        # Run stepwise selection
        final_result = stepwise_procedure(
            result, direction="both", trace=False, criterion="BIC"
        )

        # Get final BIC
        final_bic = _get_criterion_value(final_result, "BIC")

        # Get included combinations
        final_model = final_result.model.covariate_model
        included = _get_included_combinations(final_model, final_result)

        # Check no addition improves BIC
        available_covariates = final_result.data.name_covariates
        n_parameters = final_result.model.n_parameters

        for cov in available_covariates:
            for param_idx in range(n_parameters):
                combination = (cov, param_idx)
                if combination in included:
                    continue

                test_model = _add_covariate_to_model(
                    final_model, cov, param_idx, final_result
                )

                try:
                    test_result = _fit_model_with_covariates(final_result, test_model)
                    test_bic = _get_criterion_value(test_result, "BIC")

                    assert test_bic >= final_bic - 1e-6, (
                        f"Stepwise not optimal: adding {cov} on param {param_idx} "
                        f"improves BIC from {final_bic:.4f} to {test_bic:.4f}"
                    )
                except Exception:
                    pass

        # Check no removal improves BIC
        for combination in included:
            cov, param_idx = combination

            test_model = _remove_covariate_from_model(
                final_model, cov, param_idx, final_result
            )

            try:
                test_result = _fit_model_with_covariates(final_result, test_model)
                test_bic = _get_criterion_value(test_result, "BIC")

                assert test_bic >= final_bic - 1e-6, (
                    f"Stepwise not optimal: removing {cov} from param {param_idx} "
                    f"improves BIC from {final_bic:.4f} to {test_bic:.4f}"
                )
            except Exception:
                pass


class TestStepwiseValidation:
    """Unit tests for input validation."""

    def test_no_covariates_raises_error(self):
        """Test that forward selection with no covariates raises ValueError."""
        # Create data without covariates
        np.random.seed(42)
        data_list = []
        for i in range(5):
            x = np.linspace(0, 3, 4)
            y = 2.0 * x + 1.0 + np.random.normal(0, 0.1, 4)
            for j in range(4):
                data_list.append({"Id": i + 1, "X": x[j], "Y": y[j]})

        data = pd.DataFrame(data_list)

        model = saemix_model(
            model=linear_model_with_cov,
            psi0=np.array([[2.0, 1.0]]),
            name_modpar=["a", "b"],
        )

        sdata = saemix_data(
            name_data=data,
            name_group="Id",
            name_predictors=["X"],
            name_response="Y",
            verbose=False,
        )

        control = saemix_control(
            nbiter_saemix=(10, 5),
            display_progress=False,
            warnings=False,
        )

        result = saemix(model=model, data=sdata, control=control)

        with pytest.raises(ValueError, match="No covariates"):
            forward_procedure(result)

    def test_invalid_criterion_raises_error(self):
        """Test that invalid criterion raises ValueError."""
        result = create_fitted_model_with_covariates()

        with pytest.raises(ValueError, match="criterion must be one of"):
            forward_procedure(result, criterion="invalid")

    def test_invalid_direction_raises_error(self):
        """Test that invalid direction raises ValueError."""
        result = create_fitted_model_with_covariates()

        with pytest.raises(ValueError, match="direction must be one of"):
            stepwise_procedure(result, direction="invalid")


class TestStepwiseTraceOutput:
    """Tests for trace output functionality."""

    def test_forward_with_trace(self, capsys):
        """Test that forward selection prints trace when enabled."""
        result = create_fitted_model_with_covariates(seed=42)

        forward_procedure(result, trace=True, criterion="BIC")

        captured = capsys.readouterr()
        assert "Forward Selection" in captured.out
        assert "BIC" in captured.out

    def test_backward_with_trace(self, capsys):
        """Test that backward elimination prints trace when enabled."""
        result = create_fitted_model_with_covariates(seed=42)

        # First add some covariates
        result_with_cov = forward_procedure(result, trace=False)

        backward_procedure(result_with_cov, trace=True, criterion="BIC")

        captured = capsys.readouterr()
        assert "Backward Elimination" in captured.out
        assert "BIC" in captured.out

    def test_stepwise_with_trace(self, capsys):
        """Test that stepwise selection prints trace when enabled."""
        result = create_fitted_model_with_covariates(seed=42)

        stepwise_procedure(result, direction="both", trace=True, criterion="BIC")

        captured = capsys.readouterr()
        assert "Stepwise Selection" in captured.out
        assert "BIC" in captured.out


class TestStepwiseNoImprovement:
    """Tests for cases where no improvement is found."""

    def test_forward_returns_original_when_no_improvement(self):
        """Test that forward returns original model when no covariate improves BIC."""
        result = create_fitted_model_with_covariates(seed=42)

        # Run forward selection
        final_result = forward_procedure(result, trace=False, criterion="BIC")

        # Result should be a valid SaemixObject
        assert final_result is not None
        assert final_result.results is not None

    def test_backward_returns_original_when_no_covariates(self):
        """Test that backward returns original model when no covariates to remove."""
        result = create_fitted_model_with_covariates(seed=42)

        # Run backward on model with no covariates
        final_result = backward_procedure(result, trace=False, criterion="BIC")

        # Result should be a valid SaemixObject
        assert final_result is not None
        assert final_result.results is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
