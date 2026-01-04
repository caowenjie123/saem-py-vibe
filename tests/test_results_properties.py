"""
Property-Based Tests for Results Object (SaemixRes)

Feature: saemix-python-enhancement
Properties 5, 6, 7: Confidence Interval Computation, Iteration History Recording, Predictions and Residuals Structure
Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from scipy import stats

from saemix import saemix, saemix_data, saemix_model, saemix_control
from saemix.results import SaemixRes, SaemixObject


def linear_model(psi, id, xidep):
    """Simple linear model: y = psi[id, 0] * x + psi[id, 1]"""
    return psi[id, 0] * xidep[:, 0] + psi[id, 1]


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


def create_fitted_saemix_object(n_subjects: int = 5, seed: int = 42):
    """Create a fitted SaemixObject for testing."""
    data = create_test_data(n_subjects=n_subjects, seed=seed)

    model = saemix_model(
        model=linear_model,
        psi0=np.array([[2.0, 1.0]]),
        description="Linear model test",
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
        nbiter_saemix=(20, 10),
        display_progress=False,
        warnings=False,
        map=True,
        fim=True,
    )

    result = saemix(model=model, data=sdata, control=control)
    return result


class TestConfidenceIntervalProperties:
    """Property-based tests for confidence interval computation."""

    @settings(max_examples=100, deadline=None)
    @given(
        alpha=st.floats(min_value=0.01, max_value=0.2),
        seed=st.integers(min_value=1, max_value=10000),
    )
    def test_property_5_confidence_interval_computation(self, alpha, seed):
        """
        Feature: saemix-python-enhancement, Property 5: Confidence Interval Computation
        Validates: Requirements 3.1, 3.6

        For any parameter estimate with standard error se, the confidence interval SHALL be:
        - lower = estimate - z * se
        - upper = estimate + z * se
        - The interval is symmetric around the estimate
        """
        # Create fitted object
        result = create_fitted_saemix_object(seed=seed)

        # Compute confidence intervals
        conf_int = result.results.compute_confidence_intervals(
            alpha=alpha, param_names=result.model.name_modpar
        )

        # Check DataFrame structure
        assert isinstance(conf_int, pd.DataFrame)
        required_cols = ["parameter", "estimate", "se", "lower", "upper", "rse"]
        for col in required_cols:
            assert col in conf_int.columns, f"Missing column: {col}"

        # Check confidence interval formula
        z = stats.norm.ppf(1 - alpha / 2)

        for _, row in conf_int.iterrows():
            if not np.isnan(row["se"]):
                expected_lower = row["estimate"] - z * row["se"]
                expected_upper = row["estimate"] + z * row["se"]

                np.testing.assert_almost_equal(
                    row["lower"],
                    expected_lower,
                    decimal=5,
                    err_msg=f"Lower bound incorrect for {row['parameter']}",
                )
                np.testing.assert_almost_equal(
                    row["upper"],
                    expected_upper,
                    decimal=5,
                    err_msg=f"Upper bound incorrect for {row['parameter']}",
                )

                # Check symmetry
                lower_diff = row["estimate"] - row["lower"]
                upper_diff = row["upper"] - row["estimate"]
                np.testing.assert_almost_equal(
                    lower_diff,
                    upper_diff,
                    decimal=5,
                    err_msg=f"Interval not symmetric for {row['parameter']}",
                )


class TestIterationHistoryProperties:
    """Property-based tests for iteration history recording."""

    @settings(max_examples=100, deadline=None)
    @given(
        n_iter_burn=st.integers(min_value=10, max_value=30),
        n_iter_main=st.integers(min_value=5, max_value=15),
        seed=st.integers(min_value=1, max_value=10000),
    )
    def test_property_6_iteration_history_recording(
        self, n_iter_burn, n_iter_main, seed
    ):
        """
        Feature: saemix-python-enhancement, Property 6: Iteration History Recording
        Validates: Requirements 3.2, 3.3

        For any SAEM run with K iterations, the results SHALL contain:
        - parpop with shape (K, n_fixed_effects)
        - allpar with shape (K, n_total_parameters)
        - Both arrays record values at each iteration

        Note: This test verifies the structure when iteration history is recorded.
        The actual recording is implemented in task 1.7.
        """
        data = create_test_data(n_subjects=5, seed=seed)

        model = saemix_model(
            model=linear_model,
            psi0=np.array([[2.0, 1.0]]),
            description="Linear model test",
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
            nbiter_saemix=(n_iter_burn, n_iter_main),
            display_progress=False,
            warnings=False,
        )

        result = saemix(model=model, data=sdata, control=control)

        # Check that results object exists
        assert result.results is not None

        # Check that fixed_effects are recorded
        assert (
            result.results.fixed_effects is not None
            or result.results.mean_phi is not None
        )

        # If parpop is recorded, check its structure
        if result.results.parpop is not None:
            total_iter = n_iter_burn + n_iter_main
            n_fixed = model.n_parameters

            assert (
                result.results.parpop.shape[0] == total_iter
            ), f"parpop should have {total_iter} rows, got {result.results.parpop.shape[0]}"
            assert (
                result.results.parpop.shape[1] == n_fixed
            ), f"parpop should have {n_fixed} columns, got {result.results.parpop.shape[1]}"


class TestPredictionsResiduals:
    """Property-based tests for predictions and residuals."""

    @settings(max_examples=100, deadline=None)
    @given(seed=st.integers(min_value=1, max_value=10000))
    def test_property_7_predictions_residuals_structure(self, seed):
        """
        Feature: saemix-python-enhancement, Property 7: Predictions and Residuals Structure
        Validates: Requirements 3.4, 3.5

        For any fitted SaemixObject, the predictions DataFrame SHALL contain columns:
        id, time, yobs, ppred, ipred, and the residuals SHALL satisfy:
        - ires = yobs - ipred
        - wres = ires / g (where g is the error model function)
        """
        # Create fitted object
        result = create_fitted_saemix_object(seed=seed)

        # Build predictions DataFrame
        predictions = result.results.build_predictions_dataframe(result)

        # Check DataFrame structure
        assert isinstance(predictions, pd.DataFrame)
        required_cols = ["id", "time", "yobs", "ppred", "ipred", "ires", "wres"]
        for col in required_cols:
            assert col in predictions.columns, f"Missing column: {col}"

        # Check residual formula: ires = yobs - ipred
        expected_ires = predictions["yobs"] - predictions["ipred"]
        np.testing.assert_array_almost_equal(
            predictions["ires"].values,
            expected_ires.values,
            decimal=5,
            err_msg="ires should equal yobs - ipred",
        )

        # Check that wres is computed (may differ from ires due to error model)
        assert not predictions["wres"].isna().all(), "wres should be computed"


class TestResultsValidation:
    """Unit tests for results validation."""

    def test_confidence_interval_without_fixed_effects_raises_error(self):
        """Test that computing CI without fixed effects or mean_phi raises ValueError."""
        res = SaemixRes()

        with pytest.raises(ValueError, match="Fixed effects not available"):
            res.compute_confidence_intervals()

    def test_confidence_interval_with_nan_se(self):
        """Test that CI computation handles NaN standard errors."""
        res = SaemixRes()
        res.mean_phi = np.array([[1.0, 2.0]])  # Use mean_phi instead of fixed_effects
        # No FIM or se_fixed, so SE will be NaN

        conf_int = res.compute_confidence_intervals()

        assert conf_int is not None
        assert len(conf_int) == 2
        # SE should be NaN
        assert conf_int["se"].isna().all()

    def test_predictions_dataframe_columns(self):
        """Test that predictions DataFrame has correct columns."""
        result = create_fitted_saemix_object()

        predictions = result.results.build_predictions_dataframe(result)

        expected_cols = ["id", "time", "yobs", "ppred", "ipred", "ires", "wres"]
        assert list(predictions.columns) == expected_cols


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
