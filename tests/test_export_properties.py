"""
Property-Based Tests for Export Module

Feature: saemix-python-enhancement
Property 11: File Export Round-Trip
Validates: Requirements 7.1, 7.2, 7.3, 7.4
"""

import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from saemix import saemix, saemix_data, saemix_model, saemix_control
from saemix.export import save_results, export_to_csv, save_plots


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


class TestFileExportRoundTrip:
    """Property-based tests for file export round-trip."""

    @settings(max_examples=100, deadline=None)
    @given(seed=st.integers(min_value=1, max_value=10000))
    def test_property_11_file_export_round_trip(self, seed):
        """
        Feature: saemix-python-enhancement, Property 11: File Export Round-Trip
        Validates: Requirements 7.1, 7.2, 7.3, 7.4

        For any SaemixObject, saving results and then reading the saved files
        SHALL produce data equivalent to the original:
        - export_to_csv(obj, 'params.csv', 'parameters'); read_csv('params.csv')
          contains all parameter estimates
        - The directory is created if it doesn't exist
        """
        # Create fitted object
        result = create_fitted_saemix_object(seed=seed)

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test 1: Export parameters and verify round-trip
            params_file = os.path.join(tmpdir, "params.csv")
            export_to_csv(result, params_file, what="parameters")

            # Read back and verify
            params_df = pd.read_csv(params_file)

            # Check that all parameter estimates are present
            assert "parameter" in params_df.columns
            assert "estimate" in params_df.columns

            # Verify estimates match original
            res = result.results
            if res.fixed_effects is not None:
                original_estimates = res.fixed_effects
            elif res.mean_phi is not None:
                original_estimates = (
                    res.mean_phi[0] if res.mean_phi.ndim > 1 else res.mean_phi
                )
            else:
                raise ValueError("No estimates available")

            exported_estimates = params_df["estimate"].values

            np.testing.assert_array_almost_equal(
                original_estimates,
                exported_estimates,
                decimal=5,
                err_msg="Exported parameter estimates should match original",
            )

            # Test 2: Directory creation
            nested_dir = os.path.join(tmpdir, "nested", "subdir")
            nested_file = os.path.join(nested_dir, "params.csv")
            export_to_csv(result, nested_file, what="parameters")

            assert os.path.exists(nested_dir), "Directory should be created"
            assert os.path.exists(nested_file), "File should be created"

    @settings(max_examples=100, deadline=None)
    @given(seed=st.integers(min_value=1, max_value=10000))
    def test_save_results_creates_all_files(self, seed):
        """
        Test that save_results creates all expected files.
        Validates: Requirements 7.1, 7.2
        """
        result = create_fitted_saemix_object(seed=seed)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_results(result, directory=tmpdir)

            # Check that expected files are created
            expected_files = [
                "parameters.csv",
                "diagnostics.csv",
                "summary.txt",
                "omega.csv",
            ]

            for filename in expected_files:
                filepath = os.path.join(tmpdir, filename)
                assert os.path.exists(filepath), f"Expected file {filename} not found"

    @settings(max_examples=100, deadline=None)
    @given(seed=st.integers(min_value=1, max_value=10000))
    def test_export_predictions_round_trip(self, seed):
        """
        Test predictions export round-trip.
        Validates: Requirements 7.2, 7.3
        """
        result = create_fitted_saemix_object(seed=seed)

        # Build predictions first
        result.results.build_predictions_dataframe(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_file = os.path.join(tmpdir, "predictions.csv")
            export_to_csv(result, pred_file, what="predictions")

            # Read back
            pred_df = pd.read_csv(pred_file)

            # Check required columns
            required_cols = ["id", "time", "yobs", "ppred", "ipred"]
            for col in required_cols:
                assert col in pred_df.columns, f"Missing column: {col}"

            # Verify data integrity
            original_pred = result.results.predictions
            assert len(pred_df) == len(original_pred), "Row count should match"

    @settings(max_examples=100, deadline=None)
    @given(seed=st.integers(min_value=1, max_value=10000))
    def test_export_eta_round_trip(self, seed):
        """
        Test random effects (eta) export round-trip.
        Validates: Requirements 7.2, 7.3
        """
        result = create_fitted_saemix_object(seed=seed)

        with tempfile.TemporaryDirectory() as tmpdir:
            eta_file = os.path.join(tmpdir, "eta.csv")
            export_to_csv(result, eta_file, what="eta")

            # Read back
            eta_df = pd.read_csv(eta_file)

            # Check that id column exists
            assert "id" in eta_df.columns

            # Verify number of subjects
            n_subjects = result.data.n_subjects
            assert len(eta_df) == n_subjects, "Should have one row per subject"


class TestExportErrorHandling:
    """Tests for export error handling."""

    def test_export_unfitted_object_raises_error(self):
        """Test that exporting unfitted object raises ValueError."""
        from saemix.results import SaemixObject, SaemixRes
        from saemix.data import SaemixData
        from saemix.model import SaemixModel

        # Create minimal unfitted object
        data = create_test_data()
        sdata = saemix_data(
            name_data=data,
            name_group="Id",
            name_predictors=["X"],
            name_response="Y",
            verbose=False,
        )

        model = saemix_model(
            model=linear_model,
            psi0=np.array([[2.0, 1.0]]),
            description="Test",
        )

        unfitted = SaemixObject(sdata, model, {})

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="has not been fitted"):
                save_results(unfitted, directory=tmpdir)

    def test_export_invalid_what_raises_error(self):
        """Test that invalid 'what' parameter raises ValueError."""
        result = create_fitted_saemix_object()

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unknown export type"):
                export_to_csv(result, os.path.join(tmpdir, "test.csv"), what="invalid")

    def test_overwrite_false_raises_error(self):
        """Test that overwrite=False raises FileExistsError."""
        result = create_fitted_saemix_object()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "params.csv")

            # First export
            export_to_csv(result, filepath, what="parameters")

            # Second export with overwrite=False should raise
            with pytest.raises(FileExistsError):
                export_to_csv(result, filepath, what="parameters", overwrite=False)


class TestExportConfidenceIntervals:
    """Tests for confidence interval export."""

    def test_export_conf_int(self):
        """Test confidence interval export."""
        result = create_fitted_saemix_object()

        # Compute confidence intervals
        result.results.compute_confidence_intervals()

        with tempfile.TemporaryDirectory() as tmpdir:
            ci_file = os.path.join(tmpdir, "conf_int.csv")
            export_to_csv(result, ci_file, what="conf_int")

            # Read back
            ci_df = pd.read_csv(ci_file)

            # Check required columns
            required_cols = ["parameter", "estimate", "se", "lower", "upper"]
            for col in required_cols:
                assert col in ci_df.columns, f"Missing column: {col}"


class TestSavePlots:
    """Tests for plot saving functionality."""

    def test_save_plots_creates_directory(self):
        """Test that save_plots creates the output directory."""
        result = create_fitted_saemix_object()

        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")

            # This may fail for some plots if data is insufficient,
            # but should create the directory
            try:
                save_plots(result, directory=plots_dir, plots=["gof"])
            except Exception:
                pass  # Some plots may fail

            assert os.path.exists(plots_dir), "Plots directory should be created"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
