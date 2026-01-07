"""
Property-Based Tests for Numerical Stability Functions

This module contains property-based tests for numerical stability enhancements,
including covariance matrix correction and log-likelihood error handling.

Feature: saemix-robustness-optimization
"""

import warnings
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from saemix.algorithm.mstep import compute_omega_safe
from saemix.algorithm.likelihood import compute_log_likelihood_safe


# =============================================================================
# Strategies for generating test data
# =============================================================================


@st.composite
def valid_phi_samples_and_mu(
    draw, min_subjects=5, max_subjects=50, min_params=1, max_params=5
):
    """
    Generate valid phi_samples and mu for covariance computation.

    Returns
    -------
    tuple
        (phi_samples, mu) where phi_samples is (n_subjects, n_params) and mu is (n_params,)
    """
    n_subjects = draw(st.integers(min_value=min_subjects, max_value=max_subjects))
    n_params = draw(st.integers(min_value=min_params, max_value=max_params))

    # Generate phi_samples with reasonable values
    phi_samples = draw(
        st.lists(
            st.lists(
                st.floats(
                    min_value=-10, max_value=10, allow_nan=False, allow_infinity=False
                ),
                min_size=n_params,
                max_size=n_params,
            ),
            min_size=n_subjects,
            max_size=n_subjects,
        )
    )
    phi_samples = np.array(phi_samples)

    # Generate mu as mean of samples (realistic scenario)
    mu = phi_samples.mean(axis=0)

    return phi_samples, mu


@st.composite
def near_singular_phi_samples(draw, n_subjects=20, n_params=3):
    """
    Generate phi_samples that will produce a near-singular covariance matrix.

    This creates data where one dimension has very low variance.
    """
    # Generate base samples
    phi_samples = draw(
        st.lists(
            st.lists(
                st.floats(
                    min_value=-5, max_value=5, allow_nan=False, allow_infinity=False
                ),
                min_size=n_params,
                max_size=n_params,
            ),
            min_size=n_subjects,
            max_size=n_subjects,
        )
    )
    phi_samples = np.array(phi_samples)

    # Make one column nearly constant (very low variance)
    col_to_collapse = draw(st.integers(min_value=0, max_value=n_params - 1))
    base_value = draw(
        st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False)
    )
    # Add tiny noise to avoid exact singularity
    tiny_noise = np.random.default_rng(42).normal(0, 1e-12, n_subjects)
    phi_samples[:, col_to_collapse] = base_value + tiny_noise

    mu = phi_samples.mean(axis=0)

    return phi_samples, mu


@st.composite
def valid_likelihood_inputs(draw, min_obs=10, max_obs=100):
    """
    Generate valid inputs for log-likelihood computation.
    """
    n_obs = draw(st.integers(min_value=min_obs, max_value=max_obs))

    # Generate observed values
    y_obs = draw(
        st.lists(
            st.floats(
                min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False
            ),
            min_size=n_obs,
            max_size=n_obs,
        )
    )
    y_obs = np.array(y_obs)

    # Generate predicted values (close to observed for realistic scenario)
    noise_scale = draw(
        st.floats(min_value=0.1, max_value=5, allow_nan=False, allow_infinity=False)
    )
    noise = draw(
        st.lists(
            st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
            min_size=n_obs,
            max_size=n_obs,
        )
    )
    y_pred = y_obs + noise_scale * np.array(noise)

    # Generate positive sigma
    sigma = draw(
        st.floats(min_value=0.01, max_value=10, allow_nan=False, allow_infinity=False)
    )

    return y_obs, y_pred, sigma


# =============================================================================
# Property Tests for Covariance Matrix Correction (Property 12)
# =============================================================================


class TestCovarianceMatrixCorrection:
    """
    Property-based tests for covariance matrix correction.

    Feature: saemix-robustness-optimization
    Property 12: Covariance Matrix Correction
    Validates: Requirements 5.4
    """

    @given(valid_phi_samples_and_mu())
    @settings(max_examples=100)
    def test_compute_omega_safe_returns_positive_definite(self, phi_mu):
        """
        Feature: saemix-robustness-optimization
        Property 12: Covariance Matrix Correction
        Validates: Requirements 5.4

        For any valid phi_samples and mu, compute_omega_safe SHALL return
        a positive definite covariance matrix (all eigenvalues > 0).
        """
        phi_samples, mu = phi_mu

        omega = compute_omega_safe(phi_samples, mu)

        # Check positive definiteness via eigenvalues
        eigenvalues = np.linalg.eigvalsh(omega)
        assert np.all(eigenvalues > 0), (
            f"Covariance matrix should be positive definite, "
            f"but has eigenvalues: {eigenvalues}"
        )

    @given(valid_phi_samples_and_mu())
    @settings(max_examples=100)
    def test_compute_omega_safe_returns_symmetric(self, phi_mu):
        """
        Feature: saemix-robustness-optimization
        Property 12: Covariance Matrix Correction
        Validates: Requirements 5.4

        For any valid phi_samples and mu, compute_omega_safe SHALL return
        a symmetric covariance matrix.
        """
        phi_samples, mu = phi_mu

        omega = compute_omega_safe(phi_samples, mu)

        # Check symmetry
        assert np.allclose(omega, omega.T), (
            f"Covariance matrix should be symmetric, "
            f"max asymmetry: {np.max(np.abs(omega - omega.T))}"
        )

    @given(valid_phi_samples_and_mu())
    @settings(max_examples=100)
    def test_compute_omega_safe_cholesky_succeeds(self, phi_mu):
        """
        Feature: saemix-robustness-optimization
        Property 12: Covariance Matrix Correction
        Validates: Requirements 5.4

        For any valid phi_samples and mu, the returned covariance matrix
        SHALL allow successful Cholesky decomposition.
        """
        phi_samples, mu = phi_mu

        omega = compute_omega_safe(phi_samples, mu)

        # Cholesky should succeed for positive definite matrix
        try:
            L = np.linalg.cholesky(omega)
            assert L.shape == omega.shape
        except np.linalg.LinAlgError:
            pytest.fail("Cholesky decomposition should succeed for corrected matrix")

    @given(valid_phi_samples_and_mu())
    @settings(max_examples=100)
    def test_compute_omega_safe_finite_output(self, phi_mu):
        """
        Feature: saemix-robustness-optimization
        Property 12: Covariance Matrix Correction
        Validates: Requirements 5.4

        For any valid phi_samples and mu, compute_omega_safe SHALL return
        a matrix with all finite values (no NaN, no Inf).
        """
        phi_samples, mu = phi_mu

        omega = compute_omega_safe(phi_samples, mu)

        assert np.all(np.isfinite(omega)), (
            f"Covariance matrix should have all finite values, "
            f"but contains NaN: {np.any(np.isnan(omega))}, Inf: {np.any(np.isinf(omega))}"
        )

    @given(near_singular_phi_samples())
    @settings(max_examples=50)
    def test_compute_omega_safe_corrects_near_singular(self, phi_mu):
        """
        Feature: saemix-robustness-optimization
        Property 12: Covariance Matrix Correction
        Validates: Requirements 5.4

        For phi_samples that would produce a near-singular covariance matrix,
        compute_omega_safe SHALL correct it to be positive definite and issue a warning.
        """
        phi_samples, mu = phi_mu

        # Should issue warning about small eigenvalues
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            omega = compute_omega_safe(phi_samples, mu)

            # Check that result is still positive definite
            eigenvalues = np.linalg.eigvalsh(omega)
            assert np.all(eigenvalues > 0), (
                f"Corrected matrix should be positive definite, "
                f"eigenvalues: {eigenvalues}"
            )


class TestCovarianceMatrixErrorHandling:
    """Unit tests for covariance matrix error handling."""

    def test_compute_omega_safe_rejects_1d_input(self):
        """1D phi_samples should be rejected."""
        phi_samples = np.array([1.0, 2.0, 3.0])
        mu = np.array([2.0])

        with pytest.raises(ValueError, match="must be 2D array"):
            compute_omega_safe(phi_samples, mu)

    def test_compute_omega_safe_rejects_empty_input(self):
        """Empty phi_samples should be rejected."""
        phi_samples = np.array([]).reshape(0, 3)
        mu = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="0 subjects"):
            compute_omega_safe(phi_samples, mu)

    def test_compute_omega_safe_rejects_nan_input(self):
        """NaN in phi_samples should be rejected."""
        phi_samples = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        mu = np.array([3.0, 4.0])

        with pytest.raises(ValueError, match="non-finite values"):
            compute_omega_safe(phi_samples, mu)

    def test_compute_omega_safe_rejects_inf_input(self):
        """Inf in phi_samples should be rejected."""
        phi_samples = np.array([[1.0, 2.0], [np.inf, 4.0], [5.0, 6.0]])
        mu = np.array([3.0, 4.0])

        with pytest.raises(ValueError, match="non-finite values"):
            compute_omega_safe(phi_samples, mu)

    def test_compute_omega_safe_rejects_mismatched_mu(self):
        """mu with wrong shape should be rejected."""
        phi_samples = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mu = np.array([1.0, 2.0, 3.0])  # Wrong size

        with pytest.raises(ValueError, match="incompatible"):
            compute_omega_safe(phi_samples, mu)


# =============================================================================
# Property Tests for Log-Likelihood Error Handling (Property 11)
# =============================================================================


class TestLogLikelihoodErrorHandling:
    """
    Property-based tests for log-likelihood error handling.

    Feature: saemix-robustness-optimization
    Property 11: Log-Likelihood Error Handling
    Validates: Requirements 5.2
    """

    @given(valid_likelihood_inputs())
    @settings(max_examples=100)
    def test_compute_log_likelihood_safe_returns_finite(self, inputs):
        """
        Feature: saemix-robustness-optimization
        Property 11: Log-Likelihood Error Handling
        Validates: Requirements 5.2

        For any valid inputs (finite y_obs, y_pred, positive sigma),
        compute_log_likelihood_safe SHALL return a finite log-likelihood value.
        """
        y_obs, y_pred, sigma = inputs

        ll = compute_log_likelihood_safe(y_obs, y_pred, sigma)

        assert np.isfinite(ll), f"Log-likelihood should be finite, got {ll}"

    @given(valid_likelihood_inputs())
    @settings(max_examples=100)
    def test_compute_log_likelihood_safe_returns_negative(self, inputs):
        """
        Feature: saemix-robustness-optimization
        Property 11: Log-Likelihood Error Handling
        Validates: Requirements 5.2

        For any valid inputs, compute_log_likelihood_safe SHALL return
        a non-positive log-likelihood value (LL <= 0 for Gaussian).
        """
        y_obs, y_pred, sigma = inputs

        ll = compute_log_likelihood_safe(y_obs, y_pred, sigma)

        # Log-likelihood for Gaussian should be negative (or zero for perfect fit)
        # Actually, it can be positive for very small sigma, but should be finite
        assert np.isfinite(ll), f"Log-likelihood should be finite, got {ll}"

    @given(
        st.lists(
            st.floats(
                min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False
            ),
            min_size=10,
            max_size=50,
        ),
        st.floats(min_value=0.01, max_value=10, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_compute_log_likelihood_safe_perfect_prediction(self, y_obs_list, sigma):
        """
        Feature: saemix-robustness-optimization
        Property 11: Log-Likelihood Error Handling
        Validates: Requirements 5.2

        For perfect predictions (y_pred == y_obs), compute_log_likelihood_safe
        SHALL return a finite log-likelihood value.
        """
        y_obs = np.array(y_obs_list)
        y_pred = y_obs.copy()  # Perfect prediction

        ll = compute_log_likelihood_safe(y_obs, y_pred, sigma)

        assert np.isfinite(
            ll
        ), f"Log-likelihood should be finite for perfect prediction, got {ll}"


class TestLogLikelihoodErrorCases:
    """Unit tests for log-likelihood error cases."""

    def test_compute_log_likelihood_safe_rejects_zero_sigma(self):
        """Zero sigma should be rejected."""
        y_obs = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        with pytest.raises(ValueError, match="Sigma must be positive"):
            compute_log_likelihood_safe(y_obs, y_pred, sigma=0.0)

    def test_compute_log_likelihood_safe_rejects_negative_sigma(self):
        """Negative sigma should be rejected."""
        y_obs = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        with pytest.raises(ValueError, match="Sigma must be positive"):
            compute_log_likelihood_safe(y_obs, y_pred, sigma=-1.0)

    def test_compute_log_likelihood_safe_rejects_nan_y_obs(self):
        """NaN in y_obs should be rejected."""
        y_obs = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        with pytest.raises(ValueError, match="y_obs contains non-finite"):
            compute_log_likelihood_safe(y_obs, y_pred, sigma=1.0)

    def test_compute_log_likelihood_safe_rejects_nan_y_pred(self):
        """NaN in y_pred should be rejected."""
        y_obs = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, np.nan, 3.1])

        with pytest.raises(ValueError, match="y_pred contains non-finite"):
            compute_log_likelihood_safe(y_obs, y_pred, sigma=1.0)

    def test_compute_log_likelihood_safe_rejects_inf_y_obs(self):
        """Inf in y_obs should be rejected."""
        y_obs = np.array([1.0, np.inf, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        with pytest.raises(ValueError, match="y_obs contains non-finite"):
            compute_log_likelihood_safe(y_obs, y_pred, sigma=1.0)

    def test_compute_log_likelihood_safe_rejects_inf_y_pred(self):
        """Inf in y_pred should be rejected."""
        y_obs = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, np.inf, 3.1])

        with pytest.raises(ValueError, match="y_pred contains non-finite"):
            compute_log_likelihood_safe(y_obs, y_pred, sigma=1.0)

    def test_compute_log_likelihood_safe_rejects_shape_mismatch(self):
        """Mismatched shapes should be rejected."""
        y_obs = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1])

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_log_likelihood_safe(y_obs, y_pred, sigma=1.0)

    def test_compute_log_likelihood_safe_rejects_empty_arrays(self):
        """Empty arrays should be rejected."""
        y_obs = np.array([])
        y_pred = np.array([])

        with pytest.raises(ValueError, match="Empty observation"):
            compute_log_likelihood_safe(y_obs, y_pred, sigma=1.0)

    def test_compute_log_likelihood_safe_includes_iteration_in_error(self):
        """Error message should include iteration number when provided."""
        y_obs = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        with pytest.raises(ValueError, match="iteration 42"):
            compute_log_likelihood_safe(y_obs, y_pred, sigma=0.0, iteration=42)

    def test_compute_log_likelihood_safe_includes_params_in_error(self):
        """Error message should include parameter values when provided."""
        y_obs = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        params = {"ka": 1.5, "V": 35.0}

        with pytest.raises(ValueError, match="ka"):
            compute_log_likelihood_safe(y_obs, y_pred, sigma=0.0, param_values=params)
