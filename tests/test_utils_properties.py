"""
Property-Based Tests for Utils Functions

This module contains property-based tests for utility functions,
particularly the ID conversion functions.

Feature: saemix-robustness-optimization
"""

import pytest
from hypothesis import given, strategies as st, settings

from saemix.utils import id_to_index, index_to_id


class TestIDConversionProperties:
    """Property-based tests for ID conversion functions."""

    @given(st.integers(min_value=1, max_value=10000))
    @settings(max_examples=100)
    def test_id_to_index_round_trip(self, user_id):
        """
        Feature: saemix-robustness-optimization
        Property 13: ID Conversion Round-Trip
        Validates: Requirements 7.4

        For any valid user-facing 1-based ID, converting to internal 0-based
        index and back SHALL produce the original ID.
        """
        index = id_to_index(user_id)
        recovered_id = index_to_id(index)
        assert recovered_id == user_id

    @given(st.integers(min_value=0, max_value=9999))
    @settings(max_examples=100)
    def test_index_to_id_round_trip(self, index):
        """
        Feature: saemix-robustness-optimization
        Property 13: ID Conversion Round-Trip
        Validates: Requirements 7.4

        For any valid 0-based index, converting to 1-based ID and back
        SHALL produce the original index.
        """
        user_id = index_to_id(index)
        recovered_index = id_to_index(user_id)
        assert recovered_index == index

    @given(st.integers(min_value=1, max_value=10000))
    @settings(max_examples=100)
    def test_id_to_index_produces_valid_index(self, user_id):
        """
        Feature: saemix-robustness-optimization
        Property 13: ID Conversion Round-Trip
        Validates: Requirements 7.4

        For any valid 1-based ID, the resulting index SHALL be non-negative.
        """
        index = id_to_index(user_id)
        assert index >= 0
        assert index == user_id - 1

    @given(st.integers(min_value=0, max_value=9999))
    @settings(max_examples=100)
    def test_index_to_id_produces_valid_id(self, index):
        """
        Feature: saemix-robustness-optimization
        Property 13: ID Conversion Round-Trip
        Validates: Requirements 7.4

        For any valid 0-based index, the resulting ID SHALL be >= 1.
        """
        user_id = index_to_id(index)
        assert user_id >= 1
        assert user_id == index + 1


class TestIDConversionErrorHandling:
    """Unit tests for ID conversion error handling."""

    def test_id_to_index_rejects_zero(self):
        """ID 0 should be rejected as invalid."""
        with pytest.raises(ValueError, match="User ID must be >= 1"):
            id_to_index(0)

    def test_id_to_index_rejects_negative(self):
        """Negative IDs should be rejected."""
        with pytest.raises(ValueError, match="User ID must be >= 1"):
            id_to_index(-1)

        with pytest.raises(ValueError, match="User ID must be >= 1"):
            id_to_index(-100)

    def test_index_to_id_rejects_negative(self):
        """Negative indices should be rejected."""
        with pytest.raises(ValueError, match="Index must be >= 0"):
            index_to_id(-1)

        with pytest.raises(ValueError, match="Index must be >= 0"):
            index_to_id(-100)

    def test_index_to_id_accepts_zero(self):
        """Index 0 should be valid and return ID 1."""
        assert index_to_id(0) == 1


import numpy as np
from saemix.utils import transphi, transpsi, LOG_EPS, LOGIT_EPS


class TestTransformationOutputFiniteness:
    """
    Property-based tests for transformation output finiteness.

    Feature: saemix-robustness-optimization
    Property 7: Transformation Output Finiteness
    Validates: Requirements 4.1, 4.2
    """

    @given(
        st.lists(
            st.floats(
                min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_transphi_log_produces_finite_output(self, values):
        """
        Feature: saemix-robustness-optimization
        Property 7: Transformation Output Finiteness
        Validates: Requirements 4.1, 4.2

        For any finite input array with log transformation (tr=1),
        transphi SHALL produce finite output values (no NaN, no -Inf).
        Note: Very large inputs may cause overflow (+Inf), which raises ValueError.
        """
        phi = np.array(values).reshape(1, -1)
        tr = np.ones(len(values), dtype=int)

        # For reasonable inputs, output should be finite
        # Very large inputs (>700) will cause exp overflow
        max_safe_input = 700
        if np.all(np.abs(phi) <= max_safe_input):
            psi = transphi(phi, tr)
            assert np.all(
                np.isfinite(psi)
            ), "transphi with log transform should produce finite output"
            assert np.all(psi > 0), "exp() output should always be positive"

    @given(
        st.lists(
            st.floats(
                min_value=-10, max_value=10, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_transphi_probit_produces_finite_output(self, values):
        """
        Feature: saemix-robustness-optimization
        Property 7: Transformation Output Finiteness
        Validates: Requirements 4.1, 4.2

        For any finite input array with probit transformation (tr=2),
        transphi SHALL produce finite output values in [0, 1].
        """
        phi = np.array(values).reshape(1, -1)
        tr = np.full(len(values), 2, dtype=int)

        psi = transphi(phi, tr)
        assert np.all(
            np.isfinite(psi)
        ), "transphi with probit transform should produce finite output"
        assert np.all(psi >= 0) and np.all(
            psi <= 1
        ), "probit output should be in [0, 1]"

    @given(
        st.lists(
            st.floats(
                min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_transphi_logit_produces_finite_output(self, values):
        """
        Feature: saemix-robustness-optimization
        Property 7: Transformation Output Finiteness
        Validates: Requirements 4.1, 4.2

        For any finite input array with logit transformation (tr=3),
        transphi SHALL produce finite output values in [0, 1].
        """
        phi = np.array(values).reshape(1, -1)
        tr = np.full(len(values), 3, dtype=int)

        psi = transphi(phi, tr)
        assert np.all(
            np.isfinite(psi)
        ), "transphi with logit transform should produce finite output"
        assert np.all(psi >= 0) and np.all(psi <= 1), "logit output should be in [0, 1]"

    @given(
        st.lists(
            st.floats(
                min_value=LOG_EPS, max_value=1e10, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_transpsi_log_produces_finite_output(self, values):
        """
        Feature: saemix-robustness-optimization
        Property 7: Transformation Output Finiteness
        Validates: Requirements 4.1, 4.2

        For any valid input array (positive values) with log inverse transformation (tr=1),
        transpsi SHALL produce finite output values by clipping to LOG_EPS.
        """
        psi = np.array(values).reshape(1, -1)
        tr = np.ones(len(values), dtype=int)

        phi = transpsi(psi, tr)
        assert np.all(
            np.isfinite(phi)
        ), "transpsi with log inverse should produce finite output"

    @given(
        st.lists(
            st.floats(
                min_value=LOGIT_EPS,
                max_value=1 - LOGIT_EPS,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_transpsi_probit_produces_finite_output(self, values):
        """
        Feature: saemix-robustness-optimization
        Property 7: Transformation Output Finiteness
        Validates: Requirements 4.1, 4.2

        For any valid input array (values in (0,1)) with probit inverse transformation (tr=2),
        transpsi SHALL produce finite output values by clipping to (LOGIT_EPS, 1-LOGIT_EPS).
        """
        psi = np.array(values).reshape(1, -1)
        tr = np.full(len(values), 2, dtype=int)

        phi = transpsi(psi, tr)
        assert np.all(
            np.isfinite(phi)
        ), "transpsi with probit inverse should produce finite output"

    @given(
        st.lists(
            st.floats(
                min_value=LOGIT_EPS,
                max_value=1 - LOGIT_EPS,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_transpsi_logit_produces_finite_output(self, values):
        """
        Feature: saemix-robustness-optimization
        Property 7: Transformation Output Finiteness
        Validates: Requirements 4.1, 4.2

        For any valid input array (values in (0,1)) with logit inverse transformation (tr=3),
        transpsi SHALL produce finite output values by clipping to (LOGIT_EPS, 1-LOGIT_EPS).
        """
        psi = np.array(values).reshape(1, -1)
        tr = np.full(len(values), 3, dtype=int)

        phi = transpsi(psi, tr)
        assert np.all(
            np.isfinite(phi)
        ), "transpsi with logit inverse should produce finite output"


class TestInverseTransformationBounds:
    """
    Property-based tests for inverse transformation bounds.

    Feature: saemix-robustness-optimization
    Property 9: Inverse Transformation Bounds
    Validates: Requirements 4.5
    """

    @given(
        st.lists(
            st.floats(
                min_value=-1e-15, max_value=1e15, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_transpsi_log_clips_small_values(self, values):
        """
        Feature: saemix-robustness-optimization
        Property 9: Inverse Transformation Bounds
        Validates: Requirements 4.5

        For any input array with log inverse transformation (tr=1),
        transpsi SHALL clip values to LOG_EPS minimum to prevent -Inf output.
        """
        # Include some very small or zero values
        psi = np.array(values).reshape(1, -1)
        # Ensure some values are very small (near zero)
        psi = np.abs(psi)  # Make all positive
        tr = np.ones(len(values), dtype=int)

        phi = transpsi(psi, tr)
        # Output should be finite (no -Inf from log(0))
        assert np.all(
            np.isfinite(phi)
        ), "transpsi should clip small values to prevent -Inf"
        # Output should be >= log(LOG_EPS)
        assert np.all(
            phi >= np.log(LOG_EPS)
        ), f"Output should be >= log(LOG_EPS)={np.log(LOG_EPS)}"

    @given(
        st.lists(
            st.floats(
                min_value=-0.5, max_value=1.5, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_transpsi_logit_clips_boundary_values(self, values):
        """
        Feature: saemix-robustness-optimization
        Property 9: Inverse Transformation Bounds
        Validates: Requirements 4.5

        For any input array with logit inverse transformation (tr=3),
        transpsi SHALL clip values to (LOGIT_EPS, 1-LOGIT_EPS) to prevent -Inf/+Inf output.
        """
        psi = np.array(values).reshape(1, -1)
        tr = np.full(len(values), 3, dtype=int)

        phi = transpsi(psi, tr)
        # Output should be finite (no -Inf or +Inf)
        assert np.all(
            np.isfinite(phi)
        ), "transpsi should clip boundary values to prevent Inf"
        # Output should be bounded by the clipped range
        min_output = np.log(LOGIT_EPS / (1 - LOGIT_EPS))
        max_output = np.log((1 - LOGIT_EPS) / LOGIT_EPS)
        assert np.all(phi >= min_output), f"Output should be >= {min_output}"
        assert np.all(phi <= max_output), f"Output should be <= {max_output}"

    @given(
        st.lists(
            st.floats(
                min_value=-0.5, max_value=1.5, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_transpsi_probit_clips_boundary_values(self, values):
        """
        Feature: saemix-robustness-optimization
        Property 9: Inverse Transformation Bounds
        Validates: Requirements 4.5

        For any input array with probit inverse transformation (tr=2),
        transpsi SHALL clip values to (LOGIT_EPS, 1-LOGIT_EPS) to prevent -Inf/+Inf output.
        """
        psi = np.array(values).reshape(1, -1)
        tr = np.full(len(values), 2, dtype=int)

        phi = transpsi(psi, tr)
        # Output should be finite (no -Inf or +Inf)
        assert np.all(
            np.isfinite(phi)
        ), "transpsi should clip boundary values to prevent Inf"

    @given(
        st.lists(
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_transpsi_handles_exact_boundaries(self, values):
        """
        Feature: saemix-robustness-optimization
        Property 9: Inverse Transformation Bounds
        Validates: Requirements 4.5

        For input values at exact boundaries (0 and 1) with logit transformation,
        transpsi SHALL clip to valid range and produce finite output.
        """
        # Create array with some exact 0s and 1s
        psi = np.array(values).reshape(1, -1)
        tr = np.full(len(values), 3, dtype=int)

        phi = transpsi(psi, tr)
        # Should handle 0 and 1 gracefully
        assert np.all(np.isfinite(phi)), "transpsi should handle exact 0 and 1 values"
