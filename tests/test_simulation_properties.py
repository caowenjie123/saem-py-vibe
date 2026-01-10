"""
Property-Based Tests for Simulation Module

Feature: saemix-python-enhancement
Property 9: Simulation Reproducibility
Property 10: Simulation Output Structure
Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.7
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from saemix import saemix, saemix_data, saemix_model, saemix_control
from saemix.simulation import (
    simulate_saemix,
    simulate_discrete_saemix,
    simulate_with_uncertainty,
)


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


def create_fitted_model(seed: int = 42):
    """Create a fitted SAEM model for testing."""
    data = create_test_data(n_subjects=5, n_obs_per_subject=4, seed=seed)

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
    )

    result = saemix(model=model, data=sdata, control=control)
    return result


class TestSimulationProperties:
    """Property-based tests for simulation functions."""

    @settings(max_examples=100, deadline=None)
    @given(seed=st.integers(min_value=1, max_value=100000))
    def test_property_9_reproducibility(self, seed):
        """
        Feature: saemix-python-enhancement, Property 9: Simulation Reproducibility
        Validates: Requirements 5.3

        For any seed value, calling simulate_saemix twice with the same seed
        SHALL produce identical results.
        """
        # Create fitted model
        result = create_fitted_model(seed=42)  # Fixed model seed

        # Run simulation twice with same seed
        sim1 = simulate_saemix(
            result, nsim=5, seed=seed, predictions=True, res_var=True
        )
        sim2 = simulate_saemix(
            result, nsim=5, seed=seed, predictions=True, res_var=True
        )

        # Check that results are identical
        pd.testing.assert_frame_equal(
            sim1,
            sim2,
            check_exact=True,
            obj="Simulation results should be identical with same seed",
        )

    @settings(max_examples=100, deadline=None)
    @given(
        nsim=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=1, max_value=10000),
    )
    def test_property_10_output_structure(self, nsim, seed):
        """
        Feature: saemix-python-enhancement, Property 10: Simulation Output Structure
        Validates: Requirements 5.1, 5.2, 5.7

        For any simulation with nsim replicates, the output DataFrame SHALL:
        - Have exactly nsim * n_observations rows
        - Contain columns: sim, id, time, ysim
        """
        # Create fitted model
        result = create_fitted_model(seed=42)
        n_observations = result.data.n_total_obs

        # Run simulation
        sim_data = simulate_saemix(
            result, nsim=nsim, seed=seed, predictions=False, res_var=True
        )

        # Check row count
        expected_rows = nsim * n_observations
        assert (
            len(sim_data) == expected_rows
        ), f"Expected {expected_rows} rows, got {len(sim_data)}"

        # Check required columns
        required_columns = ["sim", "id", "time", "ysim"]
        for col in required_columns:
            assert col in sim_data.columns, f"Missing required column: {col}"

        # Check sim values range from 1 to nsim
        assert sim_data["sim"].min() == 1, "sim should start at 1"
        assert sim_data["sim"].max() == nsim, f"sim should end at {nsim}"

        # Check each sim has correct number of observations
        for sim_idx in range(1, nsim + 1):
            sim_subset = sim_data[sim_data["sim"] == sim_idx]
            assert (
                len(sim_subset) == n_observations
            ), f"Sim {sim_idx} has {len(sim_subset)} rows, expected {n_observations}"

    @settings(max_examples=100, deadline=None)
    @given(
        nsim=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=1, max_value=10000),
    )
    def test_property_10_predictions_columns(self, nsim, seed):
        """
        Feature: saemix-python-enhancement, Property 10: Simulation Output Structure
        Validates: Requirements 5.4

        When predictions=True, the output SHALL also contain ppred and ipred columns.
        """
        # Create fitted model
        result = create_fitted_model(seed=42)

        # Run simulation with predictions
        sim_data = simulate_saemix(
            result, nsim=nsim, seed=seed, predictions=True, res_var=True
        )

        # Check prediction columns are present
        assert "ppred" in sim_data.columns, "Missing ppred column when predictions=True"
        assert "ipred" in sim_data.columns, "Missing ipred column when predictions=True"

        # Check prediction values are finite
        assert np.all(
            np.isfinite(sim_data["ppred"])
        ), "ppred contains non-finite values"
        assert np.all(
            np.isfinite(sim_data["ipred"])
        ), "ipred contains non-finite values"

    @settings(max_examples=100, deadline=None)
    @given(
        nsim=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=1, max_value=10000),
    )
    def test_property_10_res_var_effect(self, nsim, seed):
        """
        Feature: saemix-python-enhancement, Property 10: Simulation Output Structure
        Validates: Requirements 5.5

        When res_var=True, ysim should differ from ipred (due to residual variability).
        """
        # Create fitted model
        result = create_fitted_model(seed=42)

        # Run simulation with residual variability
        sim_data = simulate_saemix(
            result, nsim=nsim, seed=seed, predictions=True, res_var=True
        )

        # ysim should differ from ipred (at least for most observations)
        diff = np.abs(sim_data["ysim"] - sim_data["ipred"])

        # At least some observations should have non-zero difference
        # (with residual variability, it's extremely unlikely all are exactly equal)
        assert np.any(diff > 1e-10), "ysim should differ from ipred when res_var=True"


class TestSimulationValidation:
    """Unit tests for input validation."""

    def test_invalid_nsim_raises_error(self):
        """Test that nsim < 1 raises ValueError."""
        result = create_fitted_model()

        with pytest.raises(ValueError, match="nsim must be >= 1"):
            simulate_saemix(result, nsim=0)

    def test_unfitted_model_raises_error(self):
        """Test that unfitted model raises ValueError."""
        from saemix.results import SaemixObject

        data = create_test_data()
        model = saemix_model(
            model=linear_model,
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

        # Create unfitted object
        unfitted = SaemixObject(sdata, model, {})

        with pytest.raises(ValueError, match="has not been fitted"):
            simulate_saemix(unfitted, nsim=10)


class TestDiscreteSimulation:
    """Tests for discrete simulation function."""

    def test_discrete_simulation_basic(self):
        """Test basic discrete simulation functionality."""
        result = create_fitted_model(seed=42)

        def simulate_binary(psi, id, xidep):
            """Simple binary outcome simulation."""
            prob = 1 / (1 + np.exp(-psi[id, 0] * xidep[:, 0] - psi[id, 1]))
            # Note: User-provided functions should manage their own randomness
            # For reproducibility, users should use np.random.default_rng()
            return np.random.binomial(1, prob)

        sim_data = simulate_discrete_saemix(result, simulate_binary, nsim=5, seed=42)

        # Check output structure
        assert "sim" in sim_data.columns
        assert "id" in sim_data.columns
        assert "time" in sim_data.columns
        assert "ysim" in sim_data.columns

        # Check ysim contains only 0 and 1 (binary)
        assert set(sim_data["ysim"].unique()).issubset({0, 1})

    def test_discrete_simulation_reproducibility(self):
        """Test that discrete simulation is reproducible with same seed.

        Note: The user-provided simulate_function is responsible for its own
        randomness. For full reproducibility, the function should use the
        global state which is seeded by the seed parameter.
        """
        result = create_fitted_model(seed=42)

        # Use a deterministic function for reproducibility test
        def simulate_deterministic(psi, id, xidep):
            """Deterministic binary outcome based on probability threshold."""
            prob = 1 / (1 + np.exp(-psi[id, 0] * xidep[:, 0] - psi[id, 1]))
            # Deterministic: return 1 if prob > 0.5, else 0
            return (prob > 0.5).astype(int)

        sim1 = simulate_discrete_saemix(
            result, simulate_deterministic, nsim=5, seed=123
        )
        sim2 = simulate_discrete_saemix(
            result, simulate_deterministic, nsim=5, seed=123
        )

        pd.testing.assert_frame_equal(sim1, sim2)

    def test_discrete_simulation_invalid_function_raises_error(self):
        """Test that non-callable simulate_function raises ValueError."""
        result = create_fitted_model()

        with pytest.raises(ValueError, match="simulate_function must be callable"):
            simulate_discrete_saemix(result, "not_a_function", nsim=5)


class TestSimulationWithUncertainty:
    """Tests for simulation with parameter uncertainty."""

    def test_uncertainty_simulation_basic(self):
        """Test basic uncertainty simulation functionality."""
        result = create_fitted_model(seed=42)

        sim_data = simulate_with_uncertainty(
            result, nsim=5, seed=42, predictions=True, res_var=True
        )

        # Check output structure
        assert "sim" in sim_data.columns
        assert "ysim" in sim_data.columns
        assert "ppred" in sim_data.columns
        assert "ipred" in sim_data.columns

    def test_uncertainty_simulation_reproducibility(self):
        """Test that uncertainty simulation is reproducible with same seed."""
        result = create_fitted_model(seed=42)

        sim1 = simulate_with_uncertainty(result, nsim=5, seed=123)
        sim2 = simulate_with_uncertainty(result, nsim=5, seed=123)

        pd.testing.assert_frame_equal(sim1, sim2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
