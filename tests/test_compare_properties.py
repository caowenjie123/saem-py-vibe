"""
Property-Based Tests for Model Comparison

Feature: saemix-python-enhancement
Property 4: Model Comparison Output Correctness
Validates: Requirements 2.1, 2.2, 2.3, 2.5
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from saemix import saemix, saemix_data, saemix_model, saemix_control
from saemix.compare import compare_saemix, aic, bic, loglik


def linear_model(psi, id, xidep):
    """Simple linear model: y = psi[id, 0] * x + psi[id, 1]"""
    return psi[id, 0] * xidep[:, 0] + psi[id, 1]


def quadratic_model(psi, id, xidep):
    """Quadratic model: y = psi[id, 0] * x^2 + psi[id, 1] * x + psi[id, 2]"""
    x = xidep[:, 0]
    return psi[id, 0] * x**2 + psi[id, 1] * x + psi[id, 2]


def create_test_data(n_subjects: int = 5, n_obs_per_subject: int = 4, seed: int = 42):
    """Create synthetic test data."""
    np.random.seed(seed)

    data_list = []
    for i in range(n_subjects):
        x = np.linspace(0, 3, n_obs_per_subject)
        true_a = 2.0 + np.random.normal(0, 0.3)
        true_b = 1.0 + np.random.normal(0, 0.2)
        y = true_a * x + true_b + np.random.normal(0, 0.1, n_obs_per_subject)

        for j in range(n_obs_per_subject):
            data_list.append({"Id": i + 1, "X": x[j], "Y": y[j]})

    return pd.DataFrame(data_list)


def create_fitted_models(seed: int = 42):
    """Create two fitted models for comparison testing."""
    data = create_test_data(n_subjects=5, n_obs_per_subject=4, seed=seed)

    # Model 1: Linear
    model1 = saemix_model(
        model=linear_model,
        psi0=np.array([[2.0, 1.0]]),
        description="Linear model",
        name_modpar=["a", "b"],
    )

    # Model 2: Linear with different initial values (same structure)
    model2 = saemix_model(
        model=linear_model,
        psi0=np.array([[1.5, 0.5]]),
        description="Linear model v2",
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
        nbiter_saemix=(20, 10), display_progress=False, warnings=False, map=True
    )

    result1 = saemix(model=model1, data=sdata, control=control)
    result2 = saemix(model=model2, data=sdata, control=control)

    return result1, result2


class TestCompareProperties:
    """Property-based tests for model comparison."""

    @settings(max_examples=100, deadline=None)
    @given(seed=st.integers(min_value=1, max_value=10000))
    def test_property_4_output_correctness(self, seed):
        """
        Feature: saemix-python-enhancement, Property 4: Model Comparison Output Correctness
        Validates: Requirements 2.1, 2.2, 2.3, 2.5

        For any set of fitted SaemixObjects, compare_saemix SHALL return a DataFrame where:
        - AIC = -2 * ll + 2 * npar for each model
        - BIC = -2 * ll + log(n_subjects) * npar for each model
        - BIC_cov = -2 * ll + log(n_total_obs) * npar for each model
        - All columns (model, npar, ll, AIC, BIC, BIC_cov) are present
        """
        # Create fitted models
        result1, result2 = create_fitted_models(seed=seed)

        # Run comparison
        comparison = compare_saemix(
            result1, result2, method="is", names=["Model1", "Model2"]
        )

        # Check all required columns are present
        required_columns = ["model", "npar", "ll", "AIC", "BIC", "BIC_cov"]
        for col in required_columns:
            assert col in comparison.columns, f"Missing column: {col}"

        # Check AIC formula: AIC = -2 * ll + 2 * npar
        for _, row in comparison.iterrows():
            expected_aic = -2 * row["ll"] + 2 * row["npar"]
            np.testing.assert_almost_equal(
                row["AIC"],
                expected_aic,
                decimal=5,
                err_msg=f"AIC formula incorrect for {row['model']}",
            )

        # Check BIC formula: BIC = -2 * ll + log(n_subjects) * npar
        n_subjects = result1.data.n_subjects
        for _, row in comparison.iterrows():
            expected_bic = -2 * row["ll"] + np.log(n_subjects) * row["npar"]
            np.testing.assert_almost_equal(
                row["BIC"],
                expected_bic,
                decimal=5,
                err_msg=f"BIC formula incorrect for {row['model']}",
            )

        # Check BIC_cov formula: BIC_cov = -2 * ll + log(n_total_obs) * npar
        n_total_obs = result1.data.n_total_obs
        for _, row in comparison.iterrows():
            expected_bic_cov = -2 * row["ll"] + np.log(n_total_obs) * row["npar"]
            np.testing.assert_almost_equal(
                row["BIC_cov"],
                expected_bic_cov,
                decimal=5,
                err_msg=f"BIC_cov formula incorrect for {row['model']}",
            )


class TestCompareValidation:
    """Unit tests for input validation."""

    def test_single_model_raises_error(self):
        """Test that single model raises ValueError."""
        result1, _ = create_fitted_models()

        with pytest.raises(ValueError, match="At least 2 models"):
            compare_saemix(result1)

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        result1, result2 = create_fitted_models()

        with pytest.raises(ValueError, match="method must be one of"):
            compare_saemix(result1, result2, method="invalid")

    def test_different_data_raises_error(self):
        """Test that models with different data raise ValueError."""
        # Create two models with different data
        data1 = create_test_data(n_subjects=5, seed=42)
        data2 = create_test_data(n_subjects=6, seed=43)  # Different number of subjects

        model = saemix_model(
            model=linear_model, psi0=np.array([[2.0, 1.0]]), name_modpar=["a", "b"]
        )

        sdata1 = saemix_data(
            name_data=data1,
            name_group="Id",
            name_predictors=["X"],
            name_response="Y",
            verbose=False,
        )

        sdata2 = saemix_data(
            name_data=data2,
            name_group="Id",
            name_predictors=["X"],
            name_response="Y",
            verbose=False,
        )

        control = saemix_control(
            nbiter_saemix=(10, 5), display_progress=False, warnings=False
        )

        result1 = saemix(model=model, data=sdata1, control=control)
        result2 = saemix(model=model, data=sdata2, control=control)

        with pytest.raises(ValueError, match="subjects"):
            compare_saemix(result1, result2)

    def test_names_length_mismatch_raises_error(self):
        """Test that mismatched names length raises ValueError."""
        result1, result2 = create_fitted_models()

        with pytest.raises(ValueError, match="Number of names"):
            compare_saemix(
                result1, result2, names=["Model1"]
            )  # Only 1 name for 2 models


class TestCompareHelperFunctions:
    """Tests for helper functions."""

    def test_aic_function(self):
        """Test the aic helper function."""
        result1, _ = create_fitted_models()

        aic_val = aic(result1, method="is")
        assert isinstance(aic_val, (int, float))
        assert np.isfinite(aic_val)

    def test_bic_function(self):
        """Test the bic helper function."""
        result1, _ = create_fitted_models()

        bic_val = bic(result1, method="is")
        assert isinstance(bic_val, (int, float))
        assert np.isfinite(bic_val)

    def test_loglik_function(self):
        """Test the loglik helper function."""
        result1, _ = create_fitted_models()

        ll_val = loglik(result1, method="is")
        assert isinstance(ll_val, (int, float))
        assert np.isfinite(ll_val)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
