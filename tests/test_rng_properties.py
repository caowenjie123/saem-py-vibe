"""
RNG Property Tests for SAEMIX

This module tests the unified random number management system.

Feature: saemix-robustness-optimization
Properties: 5, 6
Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

from saemix.control import saemix_control


class TestRNGReproducibility:
    """
    Property 5: RNG Reproducibility

    *For any* valid input data, model, and seed value, running the SAEM algorithm
    twice with identical inputs SHALL produce identical results within numerical
    precision (relative tolerance < 1e-10).

    **Validates: Requirements 3.5**
    """

    @given(seed=st.integers(min_value=1, max_value=2**31 - 1))
    @settings(max_examples=100, deadline=None)
    def test_control_rng_reproducibility_with_seed(self, seed):
        """
        Feature: saemix-robustness-optimization, Property 5: RNG Reproducibility
        Validates: Requirements 3.5

        For any seed, creating two controls with the same seed should produce
        RNGs that generate identical sequences.
        """
        # Create two controls with the same seed
        control1 = saemix_control(seed=seed, fix_seed=True)
        control2 = saemix_control(seed=seed, fix_seed=True)

        # Both should have RNG instances
        assert "rng" in control1
        assert "rng" in control2
        assert isinstance(control1["rng"], np.random.Generator)
        assert isinstance(control2["rng"], np.random.Generator)

        # Generate sequences from both RNGs
        seq1 = control1["rng"].random(100)
        seq2 = control2["rng"].random(100)

        # Sequences should be identical
        np.testing.assert_array_equal(seq1, seq2)

    @given(seed=st.integers(min_value=1, max_value=2**31 - 1))
    @settings(max_examples=100, deadline=None)
    def test_rng_parameter_takes_priority(self, seed):
        """
        Feature: saemix-robustness-optimization, Property 5: RNG Reproducibility
        Validates: Requirements 3.5

        When an RNG is provided directly, it should be used instead of creating
        a new one from the seed.
        """
        # Create an RNG
        rng = np.random.default_rng(seed)

        # Generate some values to advance the state
        expected_values = rng.random(10)

        # Create a fresh RNG with same seed
        rng_fresh = np.random.default_rng(seed)

        # Pass the fresh RNG to control
        control = saemix_control(seed=99999, rng=rng_fresh)  # Different seed

        # The control should use the provided RNG, not create a new one
        actual_values = control["rng"].random(10)

        # Values should match what we'd get from the fresh RNG
        np.testing.assert_array_equal(actual_values, expected_values)

    @given(
        seed1=st.integers(min_value=1, max_value=2**30),
        seed2=st.integers(min_value=2**30 + 1, max_value=2**31 - 1),
    )
    @settings(max_examples=50, deadline=None)
    def test_different_seeds_produce_different_sequences(self, seed1, seed2):
        """
        Feature: saemix-robustness-optimization, Property 5: RNG Reproducibility
        Validates: Requirements 3.5

        Different seeds should produce different random sequences.
        """
        assume(seed1 != seed2)

        control1 = saemix_control(seed=seed1, fix_seed=True)
        control2 = saemix_control(seed=seed2, fix_seed=True)

        seq1 = control1["rng"].random(100)
        seq2 = control2["rng"].random(100)

        # Sequences should be different
        assert not np.array_equal(seq1, seq2)


class TestGlobalRNGStatePreservation:
    """
    Property 6: Global RNG State Preservation

    *For any* global numpy random state set before calling saemix functions,
    the global state SHALL be unchanged after the function completes execution.

    **Validates: Requirements 3.3, 3.4, 3.6**
    """

    @given(
        global_seed=st.integers(min_value=1, max_value=2**31 - 1),
        control_seed=st.integers(min_value=1, max_value=2**31 - 1),
    )
    @settings(max_examples=100, deadline=None)
    def test_saemix_control_preserves_global_state(self, global_seed, control_seed):
        """
        Feature: saemix-robustness-optimization, Property 6: Global RNG State Preservation
        Validates: Requirements 3.3, 3.4, 3.6

        Creating a saemix_control should not affect the global numpy random state.
        """
        # Set global state
        np.random.seed(global_seed)

        # Record expected sequence from global state
        expected_before = np.random.rand(10)

        # Reset and create control
        np.random.seed(global_seed)
        _ = saemix_control(seed=control_seed, fix_seed=True)

        # Generate sequence after control creation
        actual_after = np.random.rand(10)

        # Global state should be unchanged
        np.testing.assert_array_equal(expected_before, actual_after)

    @given(global_seed=st.integers(min_value=1, max_value=2**31 - 1))
    @settings(max_examples=100, deadline=None)
    def test_control_rng_does_not_use_global_state(self, global_seed):
        """
        Feature: saemix-robustness-optimization, Property 6: Global RNG State Preservation
        Validates: Requirements 3.3, 3.4

        The control's RNG should be independent of the global numpy random state.
        """
        # Set global state
        np.random.seed(global_seed)
        global_seq = np.random.rand(10)

        # Create control with different seed
        control = saemix_control(seed=global_seed + 1, fix_seed=True)
        control_seq = control["rng"].random(10)

        # Sequences should be different (control uses its own RNG)
        assert not np.array_equal(global_seq, control_seq)

    @given(seed=st.integers(min_value=1, max_value=2**31 - 1))
    @settings(max_examples=50, deadline=None)
    def test_fix_seed_false_does_not_use_global_seed(self, seed):
        """
        Feature: saemix-robustness-optimization, Property 6: Global RNG State Preservation
        Validates: Requirements 3.4

        When fix_seed=False, the control should generate a random seed without
        using np.random.seed() or affecting global state.
        """
        # Set global state
        np.random.seed(seed)
        expected_global_seq = np.random.rand(10)

        # Reset global state
        np.random.seed(seed)

        # Create control with fix_seed=False
        control = saemix_control(fix_seed=False)

        # Global state should still produce the expected sequence
        actual_global_seq = np.random.rand(10)
        np.testing.assert_array_equal(expected_global_seq, actual_global_seq)

        # Control should have a valid RNG
        assert "rng" in control
        assert isinstance(control["rng"], np.random.Generator)


class TestRNGIntegration:
    """Integration tests for RNG across modules."""

    def test_control_rng_is_generator_instance(self):
        """Control should create a numpy Generator instance."""
        control = saemix_control(seed=12345)
        assert isinstance(control["rng"], np.random.Generator)

    def test_control_stores_seed(self):
        """Control should store the seed value."""
        control = saemix_control(seed=12345)
        assert control["seed"] == 12345

    def test_control_with_none_seed_creates_rng(self):
        """Control with None seed should still create an RNG."""
        control = saemix_control(seed=None, fix_seed=False)
        assert "rng" in control
        assert isinstance(control["rng"], np.random.Generator)

    @given(seed=st.integers(min_value=1, max_value=2**31 - 1))
    @settings(max_examples=20, deadline=None)
    def test_rng_multivariate_normal_reproducibility(self, seed):
        """
        Test that multivariate_normal is reproducible with same RNG.
        """
        mean = np.zeros(3)
        cov = np.eye(3)

        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)

        samples1 = rng1.multivariate_normal(mean, cov, size=10)
        samples2 = rng2.multivariate_normal(mean, cov, size=10)

        np.testing.assert_array_equal(samples1, samples2)

    @given(seed=st.integers(min_value=1, max_value=2**31 - 1))
    @settings(max_examples=20, deadline=None)
    def test_rng_standard_normal_reproducibility(self, seed):
        """
        Test that standard_normal is reproducible with same RNG.
        """
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)

        samples1 = rng1.standard_normal((10, 5))
        samples2 = rng2.standard_normal((10, 5))

        np.testing.assert_array_equal(samples1, samples2)
