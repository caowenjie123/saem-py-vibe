"""
saemix - Python implementation of the SAEM algorithm for nonlinear mixed effects models.

This package provides tools for:
- Fitting nonlinear mixed effects models using the SAEM algorithm
- Computing conditional distributions of individual parameters
- Model comparison using information criteria (AIC, BIC)
- Stepwise covariate selection
- Simulation from fitted models
- Comprehensive diagnostic plots
- Result export and visualization

Main Classes
------------
SaemixData : Data container for SAEM analysis
SaemixModel : Model specification for SAEM
SaemixObject : Main result object containing fitted model
SaemixRes : Results storage class

Main Functions
--------------
saemix : Fit a nonlinear mixed effects model using SAEM
saemix_data : Create a SaemixData object
saemix_model : Create a SaemixModel object
saemix_control : Create control options for SAEM
conddist_saemix : Compute conditional distributions
compare_saemix : Compare multiple fitted models
simulate_saemix : Simulate from fitted models

Export Functions
----------------
save_results : Save all results to directory
export_to_csv : Export specific results to CSV
save_plots : Save diagnostic plots

Plot Options
------------
PlotOptions : Configuration class for plot appearance
set_plot_options : Set global plot options
get_plot_options : Get current plot options
reset_plot_options : Reset to default options
"""

import warnings

# =============================================================================
# Core Dependencies Check (Required)
# =============================================================================
# These dependencies are required for the package to function.
# If any are missing, we raise an ImportError immediately with a clear message.

try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        f"[saemix] Missing required dependency: numpy. "
        f"Please install with: pip install numpy"
    ) from e

try:
    import pandas as pd
except ImportError as e:
    raise ImportError(
        f"[saemix] Missing required dependency: pandas. "
        f"Please install with: pip install pandas"
    ) from e

try:
    from scipy import stats
except ImportError as e:
    raise ImportError(
        f"[saemix] Missing required dependency: scipy. "
        f"Please install with: pip install scipy"
    ) from e

# =============================================================================
# Optional Dependencies Check (Lazy)
# =============================================================================
# These dependencies are checked lazily when the functionality is accessed.
# We track availability here but don't raise errors until the feature is used.

_HAS_MATPLOTLIB = False
try:
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:
    pass


def _require_matplotlib():
    """
    Check if matplotlib is available, raise ImportError if not.

    This function should be called at the start of any function that
    requires matplotlib for plotting functionality.

    Returns
    -------
    module
        The matplotlib.pyplot module if available.

    Raises
    ------
    ImportError
        If matplotlib is not installed, with a clear message on how to install it.

    Examples
    --------
    >>> def my_plot_function():
    ...     plt = _require_matplotlib()
    ...     plt.figure()
    ...     # ... plotting code ...
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "[saemix] matplotlib is required for plotting functionality. "
            "Install with: pip install matplotlib"
        )
    import matplotlib.pyplot as plt

    return plt


# =============================================================================
# Version Information
# =============================================================================
from saemix._version import __version__, __version_info__
from saemix.algorithm.conddist import compute_gelman_rubin, conddist_saemix
from saemix.algorithm.likelihood import llgq_saemix, llis_saemix
from saemix.compare import aic, bic, compare_saemix, loglik
from saemix.control import saemix_control

# =============================================================================
# Package Imports
# =============================================================================
from saemix.data import SaemixData, saemix_data
from saemix.diagnostics import (
    compute_npde,
    compute_residuals,
    npde_tests,
    plot_convergence,
    plot_correlations,
    plot_eta_distributions,
    plot_gof,
    plot_individual_fits,
    plot_likelihood,
    plot_marginal_distribution,
    plot_npde,
    plot_observed_vs_pred,
    plot_parameters_vs_covariates,
    plot_randeff_vs_covariates,
    plot_residuals,
    plot_vpc,
    simulate_observations,
)
from saemix.export import (
    export_to_csv,
    save_plots,
    save_results,
)
from saemix.main import saemix
from saemix.model import SaemixModel, saemix_model
from saemix.plot_options import (
    PlotOptions,
    apply_plot_options,
    get_plot_options,
    merge_options,
    reset_plot_options,
    set_plot_options,
)
from saemix.results import SaemixObject, SaemixRes
from saemix.simulation import (
    simulate_discrete_saemix,
    simulate_saemix,
    simulate_with_uncertainty,
)
from saemix.stepwise import (
    backward_procedure,
    forward_procedure,
    stepwise_procedure,
)

__all__ = [
    # Version info
    "__version__",
    "__version_info__",
    # Core classes
    "SaemixData",
    "saemix_data",
    "SaemixModel",
    "saemix_model",
    "saemix_control",
    "SaemixObject",
    "SaemixRes",
    "saemix",
    # Likelihood
    "llis_saemix",
    "llgq_saemix",
    # Conditional distribution
    "conddist_saemix",
    "compute_gelman_rubin",
    # Model comparison
    "compare_saemix",
    "aic",
    "bic",
    "loglik",
    # Stepwise selection
    "forward_procedure",
    "backward_procedure",
    "stepwise_procedure",
    # Simulation
    "simulate_saemix",
    "simulate_discrete_saemix",
    "simulate_with_uncertainty",
    # Diagnostics
    "plot_observed_vs_pred",
    "plot_residuals",
    "plot_individual_fits",
    "simulate_observations",
    "compute_npde",
    "npde_tests",
    "plot_npde",
    "plot_vpc",
    "compute_residuals",
    "plot_gof",
    "plot_eta_distributions",
    "plot_convergence",
    "plot_likelihood",
    "plot_parameters_vs_covariates",
    "plot_randeff_vs_covariates",
    "plot_marginal_distribution",
    "plot_correlations",
    # Export
    "save_results",
    "export_to_csv",
    "save_plots",
    # Plot options
    "PlotOptions",
    "set_plot_options",
    "get_plot_options",
    "reset_plot_options",
    "apply_plot_options",
    "merge_options",
    # Dependency helpers
    "_require_matplotlib",
    "_HAS_MATPLOTLIB",
]
