"""
Unit tests for diagnostic plotting functions.

Tests that diagnostic plot functions return correct Figure objects
and respect figure size options.

Validates: Requirements 6.7
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from saemix import saemix, saemix_data, saemix_model, saemix_control
from saemix.diagnostics import (
    plot_convergence,
    plot_likelihood,
    plot_parameters_vs_covariates,
    plot_randeff_vs_covariates,
    plot_marginal_distribution,
    plot_correlations,
)


def linear_model(psi, id, xidep):
    """Simple linear model: y = psi[id, 0] * x + psi[id, 1]"""
    return psi[id, 0] * xidep[:, 0] + psi[id, 1]


class TestDiagnosticPlots(unittest.TestCase):
    """Test diagnostic plotting functions."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - run SAEM once for all tests."""
        np.random.seed(42)

        n_subjects = 10
        n_obs_per_subject = 5

        data_list = []
        for i in range(n_subjects):
            x = np.linspace(0, 3, n_obs_per_subject)
            true_a = 2.0 + np.random.normal(0, 0.3)
            true_b = 1.0 + np.random.normal(0, 0.2)
            y = true_a * x + true_b + np.random.normal(0, 0.1, n_obs_per_subject)
            # Add a covariate (e.g., weight)
            weight = 70 + np.random.normal(0, 10)

            for j in range(n_obs_per_subject):
                data_list.append(
                    {
                        "Id": i + 1,
                        "X": x[j],
                        "Y": y[j],
                        "Weight": weight,
                    }
                )

        cls.data = pd.DataFrame(data_list)

        cls.model = saemix_model(
            model=linear_model,
            psi0=np.array([[2.0, 1.0]]),
            description="Linear model test",
            name_modpar=["a", "b"],
        )

        cls.saemix_data = saemix_data(
            name_data=cls.data,
            name_group="Id",
            name_predictors=["X"],
            name_response="Y",
            name_covariates=["Weight"],
        )

        # Run SAEM with iteration history recording
        control = saemix_control(
            nbiter_saemix=(20, 10),
            map=True,
            display_progress=False,
            warnings=False,
        )

        cls.result = saemix(
            model=cls.model,
            data=cls.saemix_data,
            control=control,
        )

    def tearDown(self):
        """Close all figures after each test."""
        plt.close("all")

    def test_plot_convergence_returns_figure(self):
        """Test that plot_convergence returns a matplotlib Figure."""
        fig = plot_convergence(self.result)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_convergence_figsize(self):
        """Test that plot_convergence respects figsize parameter."""
        figsize = (10, 6)
        fig = plot_convergence(self.result, figsize=figsize)
        self.assertIsInstance(fig, plt.Figure)
        # Check figure size (with some tolerance for DPI differences)
        actual_size = fig.get_size_inches()
        self.assertAlmostEqual(actual_size[0], figsize[0], places=1)
        self.assertAlmostEqual(actual_size[1], figsize[1], places=1)

    def test_plot_convergence_with_parameters(self):
        """Test plot_convergence with specific parameters."""
        fig = plot_convergence(self.result, parameters=["a"])
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_likelihood_returns_figure(self):
        """Test that plot_likelihood returns a matplotlib Figure."""
        fig = plot_likelihood(self.result)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_likelihood_figsize(self):
        """Test that plot_likelihood respects figsize parameter."""
        figsize = (10, 5)
        fig = plot_likelihood(self.result, figsize=figsize)
        self.assertIsInstance(fig, plt.Figure)
        actual_size = fig.get_size_inches()
        self.assertAlmostEqual(actual_size[0], figsize[0], places=1)
        self.assertAlmostEqual(actual_size[1], figsize[1], places=1)

    def test_plot_marginal_distribution_returns_figure(self):
        """Test that plot_marginal_distribution returns a matplotlib Figure."""
        fig = plot_marginal_distribution(self.result)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_marginal_distribution_figsize(self):
        """Test that plot_marginal_distribution respects figsize parameter."""
        figsize = (14, 10)
        fig = plot_marginal_distribution(self.result, figsize=figsize)
        self.assertIsInstance(fig, plt.Figure)
        actual_size = fig.get_size_inches()
        self.assertAlmostEqual(actual_size[0], figsize[0], places=1)
        self.assertAlmostEqual(actual_size[1], figsize[1], places=1)

    def test_plot_marginal_distribution_with_parameters(self):
        """Test plot_marginal_distribution with specific parameters."""
        fig = plot_marginal_distribution(self.result, parameters=["a"])
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_correlations_returns_figure(self):
        """Test that plot_correlations returns a matplotlib Figure."""
        fig = plot_correlations(self.result)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_correlations_figsize(self):
        """Test that plot_correlations respects figsize parameter."""
        figsize = (10, 10)
        fig = plot_correlations(self.result, figsize=figsize)
        self.assertIsInstance(fig, plt.Figure)
        actual_size = fig.get_size_inches()
        self.assertAlmostEqual(actual_size[0], figsize[0], places=1)
        self.assertAlmostEqual(actual_size[1], figsize[1], places=1)

    def test_plot_parameters_vs_covariates_returns_figure(self):
        """Test that plot_parameters_vs_covariates returns a matplotlib Figure."""
        fig = plot_parameters_vs_covariates(self.result)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_parameters_vs_covariates_figsize(self):
        """Test that plot_parameters_vs_covariates respects figsize parameter."""
        figsize = (14, 8)
        fig = plot_parameters_vs_covariates(self.result, figsize=figsize)
        self.assertIsInstance(fig, plt.Figure)
        actual_size = fig.get_size_inches()
        self.assertAlmostEqual(actual_size[0], figsize[0], places=1)
        self.assertAlmostEqual(actual_size[1], figsize[1], places=1)

    def test_plot_randeff_vs_covariates_returns_figure(self):
        """Test that plot_randeff_vs_covariates returns a matplotlib Figure."""
        fig = plot_randeff_vs_covariates(self.result)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_randeff_vs_covariates_figsize(self):
        """Test that plot_randeff_vs_covariates respects figsize parameter."""
        figsize = (14, 8)
        fig = plot_randeff_vs_covariates(self.result, figsize=figsize)
        self.assertIsInstance(fig, plt.Figure)
        actual_size = fig.get_size_inches()
        self.assertAlmostEqual(actual_size[0], figsize[0], places=1)
        self.assertAlmostEqual(actual_size[1], figsize[1], places=1)


class TestDiagnosticPlotsErrorHandling(unittest.TestCase):
    """Test error handling in diagnostic plotting functions."""

    def setUp(self):
        """Set up minimal test fixtures."""
        np.random.seed(42)

        # Create minimal data without covariates
        data_list = []
        for i in range(5):
            x = np.linspace(0, 3, 4)
            y = 2.0 * x + 1.0 + np.random.normal(0, 0.1, 4)
            for j in range(4):
                data_list.append({"Id": i + 1, "X": x[j], "Y": y[j]})

        self.data = pd.DataFrame(data_list)

        self.model = saemix_model(
            model=linear_model,
            psi0=np.array([[2.0, 1.0]]),
            description="Linear model test",
            name_modpar=["a", "b"],
        )

        self.saemix_data = saemix_data(
            name_data=self.data,
            name_group="Id",
            name_predictors=["X"],
            name_response="Y",
        )

        control = saemix_control(
            nbiter_saemix=(10, 5),
            map=True,
            display_progress=False,
            warnings=False,
        )

        self.result = saemix(
            model=self.model,
            data=self.saemix_data,
            control=control,
        )

    def tearDown(self):
        """Close all figures after each test."""
        plt.close("all")

    def test_plot_parameters_vs_covariates_no_covariates(self):
        """Test that plot_parameters_vs_covariates raises error when no covariates."""
        with self.assertRaises(ValueError) as context:
            plot_parameters_vs_covariates(self.result)
        self.assertIn("No covariates", str(context.exception))

    def test_plot_randeff_vs_covariates_no_covariates(self):
        """Test that plot_randeff_vs_covariates raises error when no covariates."""
        with self.assertRaises(ValueError) as context:
            plot_randeff_vs_covariates(self.result)
        self.assertIn("No covariates", str(context.exception))


if __name__ == "__main__":
    unittest.main()
