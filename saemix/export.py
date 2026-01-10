"""
Export Module

This module provides functions for saving and exporting SAEM estimation results.
Implements Requirements 7.1-7.6.
"""

import os
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from saemix.results import SaemixObject


def save_results(
    saemix_object: "SaemixObject",
    directory: str = "results",
    overwrite: bool = True,
    include_plots: bool = False,
    plot_format: str = "png",
    plot_dpi: int = 150,
) -> None:
    """
    Save all estimation results to the specified directory.

    Creates separate files for parameters, predictions, and diagnostics.

    Parameters
    ----------
    saemix_object : SaemixObject
        Fitted SAEM object with results
    directory : str
        Output directory path (default: 'results')
    overwrite : bool
        If True, overwrite existing files; if False, raise FileExistsError
    include_plots : bool
        If True, also save diagnostic plots
    plot_format : str
        Format for saved plots ('png', 'pdf', 'svg')
    plot_dpi : int
        DPI for saved plots

    Raises
    ------
    ValueError
        If saemix_object has not been fitted
    FileExistsError
        If file exists and overwrite=False
    PermissionError
        If directory cannot be created

    Notes
    -----
    Creates the following files:
    - parameters.csv: Parameter estimates with confidence intervals
    - predictions.csv: Predictions and residuals
    - diagnostics.csv: Diagnostic statistics
    - summary.txt: Text summary of results
    - omega.csv: Variance-covariance matrix of random effects
    - individual_parameters.csv: Individual parameter estimates
    """
    # Validate input
    res = saemix_object.results
    if res.fixed_effects is None and res.mean_phi is None:
        raise ValueError("SaemixObject has not been fitted. Run SAEM algorithm first.")

    # Create directory if it doesn't exist
    _ensure_directory(directory)

    # Save parameters
    _save_parameters(saemix_object, directory, overwrite)

    # Save predictions
    _save_predictions(saemix_object, directory, overwrite)

    # Save diagnostics
    _save_diagnostics(saemix_object, directory, overwrite)

    # Save summary
    _save_summary(saemix_object, directory, overwrite)

    # Save omega matrix
    _save_omega(saemix_object, directory, overwrite)

    # Save individual parameters
    _save_individual_parameters(saemix_object, directory, overwrite)

    # Save plots if requested
    if include_plots:
        plots_dir = os.path.join(directory, "plots")
        save_plots(saemix_object, plots_dir, format=plot_format, dpi=plot_dpi)


def export_to_csv(
    saemix_object: "SaemixObject",
    filename: str,
    what: str = "parameters",
    overwrite: bool = True,
) -> None:
    """
    Export specified result component to a CSV file.

    Parameters
    ----------
    saemix_object : SaemixObject
        Fitted SAEM object with results
    filename : str
        Output filename (should end with .csv)
    what : str
        Component to export: 'parameters', 'predictions', 'residuals', 'eta',
        'omega', 'individual', 'conf_int'
    overwrite : bool
        If True, overwrite existing file; if False, raise FileExistsError

    Raises
    ------
    ValueError
        If 'what' parameter is not recognized or data not available
    FileExistsError
        If file exists and overwrite=False
    """
    # Check if file exists
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(
            f"File '{filename}' already exists. Set overwrite=True to overwrite."
        )

    # Ensure parent directory exists
    parent_dir = os.path.dirname(filename)
    if parent_dir:
        _ensure_directory(parent_dir)

    res = saemix_object.results
    model = saemix_object.model
    data = saemix_object.data

    if what == "parameters":
        df = _get_parameters_df(saemix_object)
    elif what == "predictions":
        df = _get_predictions_df(saemix_object)
    elif what == "residuals":
        df = _get_residuals_df(saemix_object)
    elif what == "eta":
        df = _get_eta_df(saemix_object)
    elif what == "omega":
        df = _get_omega_df(saemix_object)
    elif what == "individual":
        df = _get_individual_params_df(saemix_object)
    elif what == "conf_int":
        df = _get_conf_int_df(saemix_object)
    else:
        raise ValueError(
            f"Unknown export type '{what}'. "
            "Valid options: 'parameters', 'predictions', 'residuals', "
            "'eta', 'omega', 'individual', 'conf_int'"
        )

    df.to_csv(filename, index=False)


def save_plots(
    saemix_object: "SaemixObject",
    directory: str = "plots",
    format: str = "png",
    dpi: int = 150,
    plots: Optional[List[str]] = None,
) -> None:
    """
    Save diagnostic plots to files.

    Parameters
    ----------
    saemix_object : SaemixObject
        Fitted SAEM object with results
    directory : str
        Output directory for plots (default: 'plots')
    format : str
        Image format: 'png', 'pdf', 'svg' (default: 'png')
    dpi : int
        Resolution in dots per inch (default: 150)
    plots : list of str, optional
        List of plot names to save. If None, saves all available plots.
        Options: 'gof', 'convergence', 'likelihood', 'eta_dist',
                 'marginal', 'correlations', 'npde', 'vpc'

    Notes
    -----
    Requires matplotlib to be installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for saving plots")

    from saemix import diagnostics

    # Create directory
    _ensure_directory(directory)

    # Define available plots
    available_plots = {
        "gof": ("gof.{ext}", diagnostics.plot_gof),
        "convergence": ("convergence.{ext}", diagnostics.plot_convergence),
        "likelihood": ("likelihood.{ext}", diagnostics.plot_likelihood),
        "eta_dist": ("eta_distributions.{ext}", diagnostics.plot_eta_distributions),
        "marginal": (
            "marginal_distribution.{ext}",
            diagnostics.plot_marginal_distribution,
        ),
        "correlations": ("correlations.{ext}", diagnostics.plot_correlations),
        "npde": ("npde.{ext}", diagnostics.plot_npde),
        "vpc": ("vpc.{ext}", diagnostics.plot_vpc),
    }

    # Determine which plots to save
    if plots is None:
        plots_to_save = list(available_plots.keys())
    else:
        plots_to_save = [p for p in plots if p in available_plots]

    # Save each plot
    for plot_name in plots_to_save:
        filename_template, plot_func = available_plots[plot_name]
        filename = os.path.join(directory, filename_template.format(ext=format))

        try:
            fig = plot_func(saemix_object)
            fig.savefig(filename, format=format, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            # Skip plots that fail (e.g., missing data)
            print(f"Warning: Could not save {plot_name} plot: {e}")
            continue


# =============================================================================
# Helper Functions
# =============================================================================


def _ensure_directory(directory: str) -> None:
    """Create directory if it doesn't exist."""
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except PermissionError:
            raise PermissionError(f"Cannot create directory '{directory}'")


def _check_overwrite(filepath: str, overwrite: bool) -> None:
    """Check if file exists and handle overwrite logic."""
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(
            f"File '{filepath}' already exists. Set overwrite=True to overwrite."
        )


def _get_parameters_df(saemix_object: "SaemixObject") -> pd.DataFrame:
    """Get parameters DataFrame."""
    res = saemix_object.results
    model = saemix_object.model

    # Get estimates - prefer fixed_effects, fall back to mean_phi
    if res.fixed_effects is not None:
        estimates = res.fixed_effects
    elif res.mean_phi is not None:
        # Use first row of mean_phi as population estimates
        estimates = res.mean_phi[0] if res.mean_phi.ndim > 1 else res.mean_phi
    else:
        raise ValueError("No parameter estimates available.")

    estimates = np.atleast_1d(estimates)
    n_params = len(estimates)

    # Get parameter names
    param_names = model.name_modpar if hasattr(model, "name_modpar") else None
    if param_names is None:
        param_names = [f"theta{i+1}" for i in range(n_params)]

    # Build DataFrame
    data = {
        "parameter": param_names[:n_params],
        "estimate": estimates,
    }

    # Add standard errors if available
    if res.se_fixed is not None:
        data["se"] = (
            res.se_fixed[:n_params]
            if len(res.se_fixed) >= n_params
            else np.concatenate(
                [res.se_fixed, np.full(n_params - len(res.se_fixed), np.nan)]
            )
        )

    # Add confidence intervals if available
    if res.conf_int is not None:
        data["lower"] = res.conf_int["lower"].values
        data["upper"] = res.conf_int["upper"].values
        if "rse" in res.conf_int.columns:
            data["rse"] = res.conf_int["rse"].values

    return pd.DataFrame(data)


def _get_predictions_df(saemix_object: "SaemixObject") -> pd.DataFrame:
    """Get predictions DataFrame."""
    res = saemix_object.results

    if res.predictions is not None:
        return res.predictions.copy()

    # Build predictions if not available
    res.build_predictions_dataframe(saemix_object)
    return res.predictions.copy() if res.predictions is not None else pd.DataFrame()


def _get_residuals_df(saemix_object: "SaemixObject") -> pd.DataFrame:
    """Get residuals DataFrame."""
    res = saemix_object.results
    data = saemix_object.data

    # Compute residuals if not available
    if res.ires is None:
        res.compute_residuals(saemix_object)

    # Build DataFrame
    df_data = {
        "id": data.data[data.name_group].values,
    }

    if res.ires is not None:
        df_data["ires"] = res.ires
    if res.wres is not None:
        df_data["wres"] = res.wres
    if res.pd_ is not None:
        df_data["pd"] = res.pd_

    return pd.DataFrame(df_data)


def _get_eta_df(saemix_object: "SaemixObject") -> pd.DataFrame:
    """Get random effects (eta) DataFrame."""
    res = saemix_object.results
    model = saemix_object.model
    data = saemix_object.data

    # Get eta values
    eta = res.map_eta if res.map_eta is not None else res.cond_mean_eta
    if eta is None:
        raise ValueError("Random effects not available. Run MAP estimation first.")

    if hasattr(eta, "values"):
        eta = eta.values

    n_eta = eta.shape[1]

    # Get parameter names
    param_names = model.name_modpar if hasattr(model, "name_modpar") else None
    if param_names is None:
        eta_names = [f"eta{i+1}" for i in range(n_eta)]
    else:
        indx_omega = model.indx_omega if hasattr(model, "indx_omega") else range(n_eta)
        eta_names = [f"eta_{param_names[i]}" for i in indx_omega[:n_eta]]

    # Get subject IDs
    subject_ids = data.data.groupby("index")[data.name_group].first().values

    # Build DataFrame
    df_data = {"id": subject_ids}
    for i, name in enumerate(eta_names):
        df_data[name] = eta[:, i]

    return pd.DataFrame(df_data)


def _get_omega_df(saemix_object: "SaemixObject") -> pd.DataFrame:
    """Get omega (variance-covariance) matrix as DataFrame."""
    res = saemix_object.results
    model = saemix_object.model

    if res.omega is None:
        raise ValueError("Omega matrix not available.")

    omega = res.omega
    n_omega = omega.shape[0]

    # Get parameter names
    param_names = model.name_modpar if hasattr(model, "name_modpar") else None
    if param_names is None:
        omega_names = [f"omega{i+1}" for i in range(n_omega)]
    else:
        indx_omega = (
            model.indx_omega if hasattr(model, "indx_omega") else range(n_omega)
        )
        omega_names = [param_names[i] for i in indx_omega[:n_omega]]

    return pd.DataFrame(omega, index=omega_names, columns=omega_names)


def _get_individual_params_df(saemix_object: "SaemixObject") -> pd.DataFrame:
    """Get individual parameter estimates DataFrame."""
    res = saemix_object.results
    model = saemix_object.model
    data = saemix_object.data

    # Get individual parameters
    phi = res.map_phi if res.map_phi is not None else res.map_psi
    if phi is None:
        phi = res.cond_mean_phi if res.cond_mean_phi is not None else res.cond_mean_psi

    if phi is None:
        raise ValueError("Individual parameters not available.")

    if hasattr(phi, "values"):
        phi = phi.values

    n_params = phi.shape[1]

    # Get parameter names
    param_names = model.name_modpar if hasattr(model, "name_modpar") else None
    if param_names is None:
        param_names = [f"theta{i+1}" for i in range(n_params)]

    # Get subject IDs
    subject_ids = data.data.groupby("index")[data.name_group].first().values

    # Build DataFrame
    df_data = {"id": subject_ids}
    for i, name in enumerate(param_names[:n_params]):
        df_data[name] = phi[:, i]

    return pd.DataFrame(df_data)


def _get_conf_int_df(saemix_object: "SaemixObject") -> pd.DataFrame:
    """Get confidence intervals DataFrame."""
    res = saemix_object.results

    if res.conf_int is not None:
        return res.conf_int.copy()

    # Compute confidence intervals if not available
    return res.compute_confidence_intervals()


def _save_parameters(
    saemix_object: "SaemixObject", directory: str, overwrite: bool
) -> None:
    """Save parameters to CSV."""
    filepath = os.path.join(directory, "parameters.csv")
    _check_overwrite(filepath, overwrite)
    df = _get_parameters_df(saemix_object)
    df.to_csv(filepath, index=False)


def _save_predictions(
    saemix_object: "SaemixObject", directory: str, overwrite: bool
) -> None:
    """Save predictions to CSV."""
    filepath = os.path.join(directory, "predictions.csv")
    _check_overwrite(filepath, overwrite)
    try:
        df = _get_predictions_df(saemix_object)
        if not df.empty:
            df.to_csv(filepath, index=False)
    except Exception:
        pass  # Skip if predictions not available


def _save_diagnostics(
    saemix_object: "SaemixObject", directory: str, overwrite: bool
) -> None:
    """Save diagnostic statistics to CSV."""
    filepath = os.path.join(directory, "diagnostics.csv")
    _check_overwrite(filepath, overwrite)

    res = saemix_object.results
    data = saemix_object.data

    # Build diagnostics DataFrame
    diagnostics = {
        "metric": [],
        "value": [],
    }

    # Add likelihood values
    if res.ll is not None:
        diagnostics["metric"].append("log_likelihood")
        diagnostics["value"].append(res.ll)
    if res.ll_is is not None:
        diagnostics["metric"].append("log_likelihood_is")
        diagnostics["value"].append(res.ll_is)
    if res.ll_gq is not None:
        diagnostics["metric"].append("log_likelihood_gq")
        diagnostics["value"].append(res.ll_gq)

    # Add information criteria
    if res.aic is not None:
        diagnostics["metric"].append("AIC")
        diagnostics["value"].append(res.aic)
    if res.bic is not None:
        diagnostics["metric"].append("BIC")
        diagnostics["value"].append(res.bic)
    if res.aic_is is not None:
        diagnostics["metric"].append("AIC_is")
        diagnostics["value"].append(res.aic_is)
    if res.bic_is is not None:
        diagnostics["metric"].append("BIC_is")
        diagnostics["value"].append(res.bic_is)

    # Add counts
    diagnostics["metric"].append("n_subjects")
    diagnostics["value"].append(data.n_subjects)
    diagnostics["metric"].append("n_observations")
    diagnostics["value"].append(data.n_total_obs)
    if res.npar_est is not None:
        diagnostics["metric"].append("n_parameters")
        diagnostics["value"].append(res.npar_est)

    df = pd.DataFrame(diagnostics)
    df.to_csv(filepath, index=False)


def _save_summary(
    saemix_object: "SaemixObject", directory: str, overwrite: bool
) -> None:
    """Save text summary to file."""
    filepath = os.path.join(directory, "summary.txt")
    _check_overwrite(filepath, overwrite)

    res = saemix_object.results
    model = saemix_object.model
    data = saemix_object.data

    lines = []
    lines.append("=" * 60)
    lines.append("SAEMIX Results Summary")
    lines.append("=" * 60)
    lines.append("")

    # Model info
    lines.append("Model Information:")
    lines.append(f"  Description: {model.description}")
    lines.append(f"  Number of subjects: {data.n_subjects}")
    lines.append(f"  Total observations: {data.n_total_obs}")
    lines.append("")

    # Fixed effects
    lines.append("Fixed Effects:")
    param_names = model.name_modpar if hasattr(model, "name_modpar") else None
    if res.fixed_effects is not None:
        n_params = len(res.fixed_effects)
        if param_names is None:
            param_names = [f"theta{i+1}" for i in range(n_params)]
        for i, (name, val) in enumerate(zip(param_names[:n_params], res.fixed_effects)):
            se_str = ""
            if res.se_fixed is not None and i < len(res.se_fixed):
                se_str = f" (SE: {res.se_fixed[i]:.4f})"
            lines.append(f"  {name}: {val:.4f}{se_str}")
    lines.append("")

    # Variance components
    lines.append("Variance Components (Omega diagonal):")
    if res.omega is not None:
        omega_diag = np.diag(res.omega)
        for i, val in enumerate(omega_diag):
            lines.append(f"  omega[{i+1},{i+1}]: {val:.4f}")
    lines.append("")

    # Residual error
    lines.append("Residual Error Parameters:")
    if res.respar is not None:
        for i, val in enumerate(res.respar):
            lines.append(f"  sigma[{i+1}]: {val:.4f}")
    lines.append("")

    # Likelihood and criteria
    lines.append("Model Fit:")
    if res.ll is not None:
        lines.append(f"  Log-likelihood: {res.ll:.4f}")
    if res.aic is not None:
        lines.append(f"  AIC: {res.aic:.4f}")
    if res.bic is not None:
        lines.append(f"  BIC: {res.bic:.4f}")
    lines.append("")

    lines.append("=" * 60)

    with open(filepath, "w") as f:
        f.write("\n".join(lines))


def _save_omega(saemix_object: "SaemixObject", directory: str, overwrite: bool) -> None:
    """Save omega matrix to CSV."""
    filepath = os.path.join(directory, "omega.csv")
    _check_overwrite(filepath, overwrite)
    try:
        df = _get_omega_df(saemix_object)
        df.to_csv(filepath)
    except Exception:
        pass  # Skip if omega not available


def _save_individual_parameters(
    saemix_object: "SaemixObject", directory: str, overwrite: bool
) -> None:
    """Save individual parameters to CSV."""
    filepath = os.path.join(directory, "individual_parameters.csv")
    _check_overwrite(filepath, overwrite)
    try:
        df = _get_individual_params_df(saemix_object)
        df.to_csv(filepath, index=False)
    except Exception:
        pass  # Skip if individual parameters not available
