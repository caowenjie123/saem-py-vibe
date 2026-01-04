"""
Property-Based Tests for Conditional Distribution Estimation

Feature: saemix-python-enhancement
Properties 1, 2, 3: MCMC Conditional Distribution Output Structure, Shrinkage Bounds, Sample Count Consistency
Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.7
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st, assume

from saemix import saemix, saemix_data, saemix_model, saemix_control
from saemix.algorithm.conddist import conddist_saemix


def linear_model(psi, id, xidep):
    """Simple linear model: y = psi[id, 0] * x + psi[id, 1]"""
    return psi[id, 0] * xidep[:, 0] + psi[id, 1]


def create_test_data(n_subjects: int, n_obs_per_subject: int, seed: int = 42):
    """Create synthetic test data for property testing."""
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


def create_fitted_saemix_object(n_subjects: int = 5, n_obs: int = 4, seed: int = 42):
    """Create a fitted SaemixObject for testing."""
    data = create_test_data(n_subjects, n_obs, seed)

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
        nbiter_saemix=(20, 10), display_progress=False, warnings=False, map=True
    )

    result = saemix(model=model, data=sdata, control=control)
    return result


class TestConddistProperties:
    """Property-based tests for conditional distribution estimation."""

    @settings(max_examples=100, deadline=None)
    @given(
        n_subjects=st.integers(min_value=3, max_value=10),
        nsamp=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=1, max_value=10000),
    )
    def test_property_1_output_structure(self, n_subjects, nsamp, seed):
        """
        Feature: saemix-python-enhancement, Property 1: MCMC Conditional Distribution Output Structure
        Validates: Requirements 1.1, 1.2, 1.7

        For any fitted SaemixObject with n_subjects and n_parameters, when conddist_saemix is called,
        the returned object SHALL have:
        - cond_mean_phi with shape (n_subjects, n_parameters)
        - cond_var_phi with shape (n_subjects, n_parameters) with all values >= 0
        - cond_shrinkage with shape (n_parameters,)
        - phi_samp with shape (n_subjects, nsamp, n_parameters)
        """
        # Create fitted object
        result = create_fitted_saemix_object(n_subjects=n_subjects, seed=seed)
        n_parameters = result.model.n_parameters

        # Run conditional distribution estimation
        result = conddist_saemix(result, nsamp=nsamp, max_iter=nsamp * 5, seed=seed)

        # Check cond_mean_phi shape
        assert result.results.cond_mean_phi is not None
        assert result.results.cond_mean_phi.shape == (
            n_subjects,
            n_parameters,
        ), f"cond_mean_phi shape {result.results.cond_mean_phi.shape} != ({n_subjects}, {n_parameters})"

        # Check cond_var_phi shape and non-negativity
        assert result.results.cond_var_phi is not None
        assert result.results.cond_var_phi.shape == (
            n_subjects,
            n_parameters,
        ), f"cond_var_phi shape {result.results.cond_var_phi.shape} != ({n_subjects}, {n_parameters})"
        assert np.all(
            result.results.cond_var_phi >= 0
        ), "cond_var_phi contains negative values"

        # Check cond_shrinkage shape
        assert result.results.cond_shrinkage is not None
        assert result.results.cond_shrinkage.shape == (
            n_parameters,
        ), f"cond_shrinkage shape {result.results.cond_shrinkage.shape} != ({n_parameters},)"

        # Check phi_samp shape
        assert result.results.phi_samp is not None
        assert result.results.phi_samp.shape == (
            n_subjects,
            nsamp,
            n_parameters,
        ), f"phi_samp shape {result.results.phi_samp.shape} != ({n_subjects}, {nsamp}, {n_parameters})"

    @settings(max_examples=100, deadline=None)
    @given(
        n_subjects=st.integers(min_value=3, max_value=10),
        seed=st.integers(min_value=1, max_value=10000),
    )
    def test_property_2_shrinkage_bounds(self, n_subjects, seed):
        """
        Feature: saemix-python-enhancement, Property 2: Shrinkage Bounds
        Validates: Requirements 1.3

        For any conditional distribution estimation result, all shrinkage values SHALL be
        between 0 and 1 (inclusive), where shrinkage = 1 - var(cond_mean) / var(population).
        """
        # Create fitted object
        result = create_fitted_saemix_object(n_subjects=n_subjects, seed=seed)

        # Run conditional distribution estimation
        result = conddist_saemix(result, nsamp=3, max_iter=30, seed=seed)

        # Check shrinkage bounds
        shrinkage = result.results.cond_shrinkage
        assert shrinkage is not None
        assert np.all(shrinkage >= 0), f"Shrinkage contains values < 0: {shrinkage}"
        assert np.all(shrinkage <= 1), f"Shrinkage contains values > 1: {shrinkage}"

    @settings(max_examples=100, deadline=None)
    @given(
        nsamp=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=1, max_value=10000),
    )
    def test_property_3_sample_count_consistency(self, nsamp, seed):
        """
        Feature: saemix-python-enhancement, Property 3: Sample Count Consistency
        Validates: Requirements 1.4

        For any valid nsamp parameter value, the phi_samp array SHALL have exactly nsamp
        samples per subject, i.e., phi_samp.shape[1] == nsamp.
        """
        # Create fitted object with fixed subjects for speed
        result = create_fitted_saemix_object(n_subjects=5, seed=seed)

        # Run conditional distribution estimation
        result = conddist_saemix(result, nsamp=nsamp, max_iter=nsamp * 5, seed=seed)

        # Check sample count
        assert result.results.phi_samp is not None
        assert (
            result.results.phi_samp.shape[1] == nsamp
        ), f"phi_samp has {result.results.phi_samp.shape[1]} samples, expected {nsamp}"


class TestConddistValidation:
    """Unit tests for input validation."""

    def test_unfitted_object_raises_error(self):
        """Test that unfitted object raises ValueError."""
        from saemix.results import SaemixObject, SaemixRes
        from saemix.data import SaemixData
        from saemix.model import SaemixModel

        data = create_test_data(5, 4)
        model = saemix_model(
            model=linear_model, psi0=np.array([[2.0, 1.0]]), name_modpar=["a", "b"]
        )
        sdata = saemix_data(
            name_data=data,
            name_group="Id",
            name_predictors=["X"],
            name_response="Y",
            verbose=False,
        )

        # Create unfitted object
        unfitted = SaemixObject(sdata, model, {})

        with pytest.raises(ValueError, match="has not been fitted"):
            conddist_saemix(unfitted)

    def test_invalid_nsamp_raises_error(self):
        """Test that nsamp < 1 raises ValueError."""
        result = create_fitted_saemix_object()

        with pytest.raises(ValueError, match="nsamp must be >= 1"):
            conddist_saemix(result, nsamp=0)

    def test_invalid_max_iter_raises_error(self):
        """Test that max_iter < nsamp raises ValueError."""
        result = create_fitted_saemix_object()

        with pytest.raises(ValueError, match="max_iter must be >= nsamp"):
            conddist_saemix(result, nsamp=10, max_iter=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
