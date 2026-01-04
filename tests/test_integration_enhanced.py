"""
Enhanced Integration Tests for Python saemix

This module tests the complete workflow of the saemix library:
- Data loading
- Model fitting
- Conditional distribution estimation
- Model comparison
- Simulation
- Result export

Uses theo.saemix.tab and cow.saemix.tab datasets from the R package.

Requirements: All
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
import pandas as pd

from saemix import (
    saemix,
    saemix_data,
    saemix_model,
    saemix_control,
    conddist_saemix,
    compare_saemix,
    simulate_saemix,
    save_results,
    export_to_csv,
    PlotOptions,
    set_plot_options,
    get_plot_options,
    reset_plot_options,
)


def model1cpt(psi, id, xidep):
    """
    One-compartment PK model with first-order absorption.

    Parameters:
    - psi: Individual parameters (ka, V, CL)
    - id: Subject indices
    - xidep: Predictors (Dose, Time)

    Returns:
    - Predicted concentrations
    """
    dose = xidep[:, 0]
    tim = xidep[:, 1]
    ka = psi[id, 0]
    V = psi[id, 1]
    CL = psi[id, 2]
    k = CL / V

    # Avoid division by zero when ka == k
    ka_safe = np.where(np.abs(ka - k) < 1e-10, ka + 1e-10, ka)

    ypred = (
        dose
        * ka_safe
        / (V * (ka_safe - k))
        * (np.exp(-k * tim) - np.exp(-ka_safe * tim))
    )
    return np.maximum(ypred, 1e-10)  # Ensure positive predictions


def growth_model(psi, id, xidep):
    """
    Asymptotic growth model for cow weight data.

    Parameters:
    - psi: Individual parameters (A, b, k)
    - id: Subject indices
    - xidep: Predictors (time)

    Returns:
    - Predicted weights
    """
    time = xidep[:, 0]
    A = psi[id, 0]  # Asymptotic weight
    b = psi[id, 1]  # Initial weight parameter
    k = psi[id, 2]  # Growth rate

    ypred = A * (1 - b * np.exp(-k * time / 1000))
    return np.maximum(ypred, 1e-10)


class TestTheoDataWorkflow(unittest.TestCase):
    """Test complete workflow with theophylline PK data."""

    @classmethod
    def setUpClass(cls):
        """Load theophylline data and set up model."""
        # Load data
        data_path = os.path.join("saemix-main", "data", "theo.saemix.tab")
        cls.theo_data = pd.read_csv(data_path, sep=" ")

        # Create saemix data object
        cls.saemix_data = saemix_data(
            name_data=cls.theo_data,
            name_group="Id",
            name_predictors=["Dose", "Time"],
            name_response="Concentration",
        )

        # Create model
        cls.model = saemix_model(
            model=model1cpt,
            description="One-compartment PK model",
            psi0=np.array([[1.5, 30.0, 2.0]]),
            name_modpar=["ka", "V", "CL"],
            transform_par=[1, 1, 1],  # Log-transform all parameters
            covariance_model=np.eye(3),
            omega_init=np.diag([0.5, 0.5, 0.5]),
            error_model="constant",
        )

        # Create temp directory for exports
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up temp directory."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_01_basic_fit(self):
        """Test basic model fitting."""
        control = saemix_control(
            nbiter_saemix=(20, 10),
            seed=12345,
            display_progress=False,
            warnings=False,
            map=True,
        )

        result = saemix(model=self.model, data=self.saemix_data, control=control)

        # Store result for subsequent tests
        self.__class__.fitted_result = result

        # Verify basic results
        self.assertIsNotNone(result.results.mean_phi)
        self.assertIsNotNone(result.results.omega)
        self.assertEqual(result.results.mean_phi.shape[0], self.saemix_data.n_subjects)
        self.assertEqual(result.results.mean_phi.shape[1], 3)  # 3 parameters

        # Verify omega is positive semi-definite
        eigenvalues = np.linalg.eigvalsh(result.results.omega)
        self.assertTrue(np.all(eigenvalues >= -1e-10))

    def test_02_conditional_distribution(self):
        """Test conditional distribution estimation."""
        if not hasattr(self.__class__, "fitted_result"):
            self.skipTest("Requires fitted result from test_01")

        result = conddist_saemix(
            self.__class__.fitted_result, nsamp=5, max_iter=50, seed=42
        )

        # Verify conditional distribution outputs
        self.assertIsNotNone(result.results.cond_mean_phi)
        self.assertIsNotNone(result.results.cond_var_phi)
        self.assertIsNotNone(result.results.cond_shrinkage)
        self.assertIsNotNone(result.results.phi_samp)

        # Check shapes
        n_subjects = self.saemix_data.n_subjects
        n_params = 3

        self.assertEqual(result.results.cond_mean_phi.shape, (n_subjects, n_params))
        self.assertEqual(result.results.cond_var_phi.shape, (n_subjects, n_params))
        self.assertEqual(result.results.cond_shrinkage.shape, (n_params,))
        self.assertEqual(result.results.phi_samp.shape, (n_subjects, 5, n_params))

        # Verify variance is non-negative
        self.assertTrue(np.all(result.results.cond_var_phi >= 0))

        # Verify shrinkage is between 0 and 1
        self.assertTrue(np.all(result.results.cond_shrinkage >= 0))
        self.assertTrue(np.all(result.results.cond_shrinkage <= 1))

    def test_03_simulation(self):
        """Test simulation from fitted model."""
        if not hasattr(self.__class__, "fitted_result"):
            self.skipTest("Requires fitted result from test_01")

        sim_data = simulate_saemix(
            self.__class__.fitted_result,
            nsim=10,
            seed=123,
            predictions=True,
            res_var=True,
        )

        # Verify simulation output structure
        self.assertIsInstance(sim_data, pd.DataFrame)
        self.assertIn("sim", sim_data.columns)
        self.assertIn("id", sim_data.columns)
        self.assertIn("time", sim_data.columns)
        self.assertIn("ysim", sim_data.columns)
        self.assertIn("ppred", sim_data.columns)
        self.assertIn("ipred", sim_data.columns)

        # Verify correct number of rows
        n_obs = len(self.theo_data)
        expected_rows = 10 * n_obs
        self.assertEqual(len(sim_data), expected_rows)

        # Verify simulation replicates
        self.assertEqual(sim_data["sim"].nunique(), 10)

    def test_04_simulation_reproducibility(self):
        """Test that simulation is reproducible with same seed."""
        if not hasattr(self.__class__, "fitted_result"):
            self.skipTest("Requires fitted result from test_01")

        sim1 = simulate_saemix(
            self.__class__.fitted_result, nsim=5, seed=999, predictions=True
        )

        sim2 = simulate_saemix(
            self.__class__.fitted_result, nsim=5, seed=999, predictions=True
        )

        # Results should be identical
        pd.testing.assert_frame_equal(sim1, sim2)

    def test_05_export_results(self):
        """Test result export functionality."""
        if not hasattr(self.__class__, "fitted_result"):
            self.skipTest("Requires fitted result from test_01")

        export_dir = os.path.join(self.temp_dir, "theo_results")

        # Save all results
        save_results(self.__class__.fitted_result, directory=export_dir, overwrite=True)

        # Verify files were created
        self.assertTrue(os.path.exists(os.path.join(export_dir, "parameters.csv")))
        self.assertTrue(os.path.exists(os.path.join(export_dir, "summary.txt")))
        self.assertTrue(os.path.exists(os.path.join(export_dir, "omega.csv")))

        # Verify parameters can be read back
        params_df = pd.read_csv(os.path.join(export_dir, "parameters.csv"))
        self.assertIn("parameter", params_df.columns)
        self.assertIn("estimate", params_df.columns)

    def test_06_export_to_csv(self):
        """Test individual CSV export."""
        if not hasattr(self.__class__, "fitted_result"):
            self.skipTest("Requires fitted result from test_01")

        # Export parameters
        params_file = os.path.join(self.temp_dir, "params_only.csv")
        export_to_csv(
            self.__class__.fitted_result, filename=params_file, what="parameters"
        )

        self.assertTrue(os.path.exists(params_file))
        params_df = pd.read_csv(params_file)
        self.assertEqual(len(params_df), 3)  # 3 parameters


class TestModelComparison(unittest.TestCase):
    """Test model comparison functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up two models for comparison."""
        # Load data
        data_path = os.path.join("saemix-main", "data", "theo.saemix.tab")
        cls.theo_data = pd.read_csv(data_path, sep=" ")

        cls.saemix_data = saemix_data(
            name_data=cls.theo_data,
            name_group="Id",
            name_predictors=["Dose", "Time"],
            name_response="Concentration",
        )

        # Model 1: Full random effects
        cls.model1 = saemix_model(
            model=model1cpt,
            description="Full model",
            psi0=np.array([[1.5, 30.0, 2.0]]),
            name_modpar=["ka", "V", "CL"],
            transform_par=[1, 1, 1],
            covariance_model=np.eye(3),
            omega_init=np.diag([0.5, 0.5, 0.5]),
            error_model="constant",
        )

        # Model 2: Reduced random effects (diagonal only for ka and V)
        cls.model2 = saemix_model(
            model=model1cpt,
            description="Reduced model",
            psi0=np.array([[1.5, 30.0, 2.0]]),
            name_modpar=["ka", "V", "CL"],
            transform_par=[1, 1, 1],
            covariance_model=np.diag([1, 1, 0]),  # No random effect on CL
            omega_init=np.diag([0.5, 0.5, 0.01]),
            error_model="constant",
        )

        # Fit both models
        control = saemix_control(
            nbiter_saemix=(15, 8), seed=54321, display_progress=False, warnings=False
        )

        cls.result1 = saemix(cls.model1, cls.saemix_data, control)
        cls.result2 = saemix(cls.model2, cls.saemix_data, control)

    def test_compare_two_models(self):
        """Test comparison of two models."""
        comparison = compare_saemix(
            self.result1, self.result2, method="is", names=["Full", "Reduced"]
        )

        # Verify output structure
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertIn("model", comparison.columns)
        self.assertIn("npar", comparison.columns)
        self.assertIn("ll", comparison.columns)
        self.assertIn("AIC", comparison.columns)
        self.assertIn("BIC", comparison.columns)
        self.assertIn("BIC_cov", comparison.columns)

        # Verify both models are in comparison
        self.assertEqual(len(comparison), 2)
        self.assertIn("Full", comparison["model"].values)
        self.assertIn("Reduced", comparison["model"].values)

        # Verify AIC formula: AIC = -2*ll + 2*npar
        for _, row in comparison.iterrows():
            expected_aic = -2 * row["ll"] + 2 * row["npar"]
            self.assertAlmostEqual(row["AIC"], expected_aic, places=4)

    def test_compare_requires_two_models(self):
        """Test that comparison requires at least 2 models."""
        with self.assertRaises(ValueError):
            compare_saemix(self.result1)

    def test_compare_invalid_method(self):
        """Test that invalid method raises error."""
        with self.assertRaises(ValueError):
            compare_saemix(self.result1, self.result2, method="invalid")


class TestCowDataWorkflow(unittest.TestCase):
    """Test workflow with cow growth data."""

    @classmethod
    def setUpClass(cls):
        """Load cow data and set up model."""
        # Load data
        data_path = os.path.join("saemix-main", "data", "cow.saemix.tab")
        cls.cow_data = pd.read_csv(data_path, sep=" ")

        # Use subset of data for faster testing
        unique_cows = cls.cow_data["cow"].unique()[:10]
        cls.cow_data_subset = cls.cow_data[cls.cow_data["cow"].isin(unique_cows)].copy()

        # Create saemix data object
        cls.saemix_data = saemix_data(
            name_data=cls.cow_data_subset,
            name_group="cow",
            name_predictors=["time"],
            name_response="weight",
        )

        # Create growth model
        cls.model = saemix_model(
            model=growth_model,
            description="Asymptotic growth model",
            psi0=np.array([[700.0, 0.9, 1.5]]),
            name_modpar=["A", "b", "k"],
            transform_par=[0, 0, 1],  # Log-transform k only
            covariance_model=np.eye(3),
            omega_init=np.diag([5000, 0.01, 0.1]),
            error_model="constant",
        )

        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up temp directory."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_01_fit_growth_model(self):
        """Test fitting growth model to cow data."""
        control = saemix_control(
            nbiter_saemix=(15, 8),
            seed=11111,
            display_progress=False,
            warnings=False,
            map=True,
        )

        result = saemix(model=self.model, data=self.saemix_data, control=control)

        self.__class__.fitted_result = result

        # Verify results
        self.assertIsNotNone(result.results.mean_phi)
        self.assertIsNotNone(result.results.omega)

        # Asymptotic weight should be reasonable (500-900 kg)
        A_estimates = result.results.mean_phi[:, 0]
        self.assertTrue(np.all(A_estimates > 400))
        self.assertTrue(np.all(A_estimates < 1000))

    def test_02_simulate_growth(self):
        """Test simulation from growth model."""
        if not hasattr(self.__class__, "fitted_result"):
            self.skipTest("Requires fitted result from test_01")

        sim_data = simulate_saemix(
            self.__class__.fitted_result, nsim=5, seed=222, predictions=True
        )

        # Verify simulation
        self.assertIsInstance(sim_data, pd.DataFrame)
        self.assertEqual(sim_data["sim"].nunique(), 5)

        # Most simulated weights should be positive (allow some noise from residual error)
        positive_ratio = np.mean(sim_data["ysim"] > 0)
        self.assertGreater(positive_ratio, 0.95)  # At least 95% positive

    def test_03_export_cow_results(self):
        """Test exporting cow model results."""
        if not hasattr(self.__class__, "fitted_result"):
            self.skipTest("Requires fitted result from test_01")

        export_dir = os.path.join(self.temp_dir, "cow_results")

        save_results(self.__class__.fitted_result, directory=export_dir, overwrite=True)

        # Verify export
        self.assertTrue(os.path.exists(os.path.join(export_dir, "parameters.csv")))

        # Read and verify parameters
        params = pd.read_csv(os.path.join(export_dir, "parameters.csv"))
        param_names = params["parameter"].tolist()
        self.assertIn("A", param_names)
        self.assertIn("b", param_names)
        self.assertIn("k", param_names)


class TestPlotOptions(unittest.TestCase):
    """Test plot options management."""

    def setUp(self):
        """Reset plot options before each test."""
        reset_plot_options()

    def tearDown(self):
        """Reset plot options after each test."""
        reset_plot_options()

    def test_default_options(self):
        """Test default plot options."""
        options = get_plot_options()

        self.assertIsInstance(options, PlotOptions)
        self.assertEqual(options.figsize, (10, 8))
        self.assertEqual(options.dpi, 100)

    def test_set_options(self):
        """Test setting plot options."""
        set_plot_options(figsize=(12, 10), dpi=150)

        options = get_plot_options()
        self.assertEqual(options.figsize, (12, 10))
        self.assertEqual(options.dpi, 150)

    def test_reset_options(self):
        """Test resetting plot options."""
        # Change options
        set_plot_options(figsize=(20, 20), dpi=300)

        # Reset
        reset_plot_options()

        # Verify defaults restored
        options = get_plot_options()
        self.assertEqual(options.figsize, (10, 8))
        self.assertEqual(options.dpi, 100)


class TestCompleteWorkflow(unittest.TestCase):
    """Test complete end-to-end workflow."""

    def test_full_workflow(self):
        """Test complete workflow: load → fit → conddist → compare → simulate → export."""
        # 1. Load data
        data_path = os.path.join("saemix-main", "data", "theo.saemix.tab")
        data = pd.read_csv(data_path, sep=" ")

        saemix_data_obj = saemix_data(
            name_data=data,
            name_group="Id",
            name_predictors=["Dose", "Time"],
            name_response="Concentration",
        )

        # 2. Create and fit model
        model = saemix_model(
            model=model1cpt,
            description="PK model",
            psi0=np.array([[1.5, 30.0, 2.0]]),
            name_modpar=["ka", "V", "CL"],
            transform_par=[1, 1, 1],
            covariance_model=np.eye(3),
            omega_init=np.diag([0.5, 0.5, 0.5]),
            error_model="constant",
        )

        control = saemix_control(
            nbiter_saemix=(15, 8),
            seed=99999,
            display_progress=False,
            warnings=False,
            map=True,
        )

        result = saemix(model, saemix_data_obj, control)

        # 3. Conditional distribution
        result = conddist_saemix(result, nsamp=3, seed=42)

        self.assertIsNotNone(result.results.cond_mean_phi)
        self.assertIsNotNone(result.results.cond_shrinkage)

        # 4. Simulation
        sim_data = simulate_saemix(result, nsim=5, seed=123)

        self.assertEqual(len(sim_data), 5 * len(data))

        # 5. Export
        with tempfile.TemporaryDirectory() as temp_dir:
            save_results(result, directory=temp_dir)

            # Verify files exist
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "parameters.csv")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "summary.txt")))


if __name__ == "__main__":
    unittest.main()
