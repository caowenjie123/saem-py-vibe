import numpy as np
from scipy.stats import chi2, norm, rankdata, shapiro, ttest_1samp

from saemix.algorithm.map_estimation import error_function


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting utilities") from exc
    return plt


def _apply_plot_options():
    """Apply global plot options if available."""
    try:
        from saemix.plot_options import apply_plot_options

        apply_plot_options()
    except ImportError:
        pass  # plot_options not available


def _get_figsize(local_figsize=None):
    """Get figure size from local or global options."""
    if local_figsize is not None:
        return local_figsize
    try:
        from saemix.plot_options import get_plot_options

        return get_plot_options().figsize
    except ImportError:
        return (10, 8)


def _get_alpha(local_alpha=None):
    """Get alpha value from local or global options."""
    if local_alpha is not None:
        return local_alpha
    try:
        from saemix.plot_options import get_plot_options

        return get_plot_options().alpha
    except ImportError:
        return 0.7


def plot_observed_vs_pred(saemix_object, pred_type="ppred", ax=None):
    """
    Scatter plot of observed vs predicted values.
    """
    plt = _require_matplotlib()
    data = saemix_object.data
    yobs = data.data[data.name_response].values
    pred = saemix_object.predict(type=pred_type)
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(pred, yobs, alpha=0.6, edgecolor="none")
    minv = min(np.min(pred), np.min(yobs))
    maxv = max(np.max(pred), np.max(yobs))
    ax.plot([minv, maxv], [minv, maxv], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Predicted ({pred_type})")
    ax.set_ylabel("Observed")
    ax.set_title("Observed vs Predicted")
    return ax


def compute_residuals(saemix_object, pred_type="ppred", standardized=True):
    """
    Compute residuals (optionally standardized by error model).
    """
    data = saemix_object.data
    model = saemix_object.model
    res = saemix_object.results
    yobs = data.data[data.name_response].values
    pred = saemix_object.predict(type=pred_type)
    if not standardized:
        return yobs - pred
    ytype = data.data["ytype"].values if "ytype" in data.data.columns else None
    ytype_norm = _normalize_ytype(ytype, len(model.error_model))
    g = error_function(pred, res.respar, model.error_model, ytype_norm)
    return (yobs - pred) / g


def plot_residuals(saemix_object, pred_type="ppred", ax=None):
    """
    Residuals vs predictor plot using the first predictor (name_X if set).
    """
    plt = _require_matplotlib()
    data = saemix_object.data
    pred = saemix_object.predict(type=pred_type)
    resid = data.data[data.name_response].values - pred
    xname = data.name_X if data.name_X else data.name_predictors[0]
    x = data.data[xname].values
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(x, resid, alpha=0.6, edgecolor="none")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(xname)
    ax.set_ylabel("Residual")
    ax.set_title(f"Residuals vs {xname}")
    return ax


def plot_individual_fits(saemix_object, pred_type="ppred", n=6):
    """
    Small multiples of observed vs predicted per subject.
    """
    plt = _require_matplotlib()
    data = saemix_object.data
    pred = saemix_object.predict(type=pred_type)
    xname = data.name_X if data.name_X else data.name_predictors[0]
    x = data.data[xname].values
    y = data.data[data.name_response].values
    idx = data.data["index"].values
    subjects = np.unique(idx)
    n = min(n, len(subjects))
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False
    )
    for k in range(n):
        ax = axes[k // ncols][k % ncols]
        sid = subjects[k]
        mask = idx == sid
        ax.scatter(x[mask], y[mask], color="black", s=15, label="obs")
        ax.plot(x[mask], pred[mask], color="tab:blue", label=pred_type)
        ax.set_title(f"Subject {sid}")
        ax.set_xlabel(xname)
        ax.set_ylabel(data.name_response)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.tight_layout()
    return fig


def plot_gof(saemix_object, pred_type="ppred"):
    """
    4-panel GOF plot: obs vs pred, residuals vs pred, residuals vs time, QQ of residuals.
    """
    plt = _require_matplotlib()
    data = saemix_object.data
    pred = saemix_object.predict(type=pred_type)
    resid = compute_residuals(saemix_object, pred_type=pred_type, standardized=True)
    yobs = data.data[data.name_response].values
    xname = data.name_X if data.name_X else data.name_predictors[0]
    x = data.data[xname].values
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.scatter(pred, yobs, alpha=0.6, edgecolor="none")
    minv = min(np.min(pred), np.min(yobs))
    maxv = max(np.max(pred), np.max(yobs))
    ax.plot([minv, maxv], [minv, maxv], color="black", linestyle="--", linewidth=1)
    ax.set_title("Observed vs Predicted")
    ax.set_xlabel(pred_type)
    ax.set_ylabel("Observed")
    ax = axes[0, 1]
    ax.scatter(pred, resid, alpha=0.6, edgecolor="none")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Std residuals vs Predicted")
    ax.set_xlabel(pred_type)
    ax.set_ylabel("Std residual")
    ax = axes[1, 0]
    ax.scatter(x, resid, alpha=0.6, edgecolor="none")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Std residuals vs {xname}")
    ax.set_xlabel(xname)
    ax.set_ylabel("Std residual")
    ax = axes[1, 1]
    qq = np.sort(resid)
    theo = norm.ppf((np.arange(1, len(qq) + 1) - 0.5) / len(qq))
    ax.scatter(theo, qq, s=10)
    ax.plot(
        [theo.min(), theo.max()],
        [theo.min(), theo.max()],
        color="black",
        linestyle="--",
    )
    ax.set_title("Residual QQ plot")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    fig.tight_layout()
    return fig


def plot_eta_distributions(saemix_object, use_map=True):
    """
    Histograms of individual random effects (eta).
    """
    plt = _require_matplotlib()
    res = saemix_object.results
    eta = res.map_eta if use_map else res.cond_mean_eta
    if eta is None:
        raise ValueError(
            "Eta not available; run MAP estimation or compute conditional means."
        )
    if hasattr(eta, "values"):
        eta = eta.values
    npar = eta.shape[1]
    ncols = int(np.ceil(np.sqrt(npar)))
    nrows = int(np.ceil(npar / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False
    )
    for j in range(npar):
        ax = axes[j // ncols][j % ncols]
        ax.hist(eta[:, j], bins=30, density=True, alpha=0.7)
        x = np.linspace(np.min(eta[:, j]), np.max(eta[:, j]), 200)
        ax.plot(x, norm.pdf(x, loc=0, scale=1), color="black", linestyle="--")
        ax.set_title(f"eta[{j}]")
    for j in range(npar, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.tight_layout()
    return fig


def simulate_observations(saemix_object, nsim=1000, pred_type="ppred", seed=None):
    """
    Simulate replicated observations using the fitted error model.
    Returns an array of shape (nsim, nobs).
    """
    rng = np.random.default_rng(seed)
    data = saemix_object.data
    model = saemix_object.model
    res = saemix_object.results
    yobs = data.data[data.name_response].values
    nobs = len(yobs)
    f = saemix_object.predict(type=pred_type)
    ytype = data.data["ytype"].values if "ytype" in data.data.columns else None
    error_model = model.error_model
    pres = res.respar
    ytype_norm = _normalize_ytype(ytype, len(error_model))
    g = error_function(f, pres, error_model, ytype_norm)
    eps = rng.standard_normal(size=(nsim, nobs))
    ysim = np.zeros((nsim, nobs))
    if len(error_model) == 1:
        em = error_model[0]
        if em == "exponential":
            a = pres[0]
            ysim = f[None, :] * np.exp(a * eps)
        else:
            ysim = f[None, :] + g[None, :] * eps
    else:
        ysim = f[None, :] + g[None, :] * eps
        for ityp, em in enumerate(error_model):
            if em != "exponential":
                continue
            mask = (
                ytype_norm == ityp
                if ytype_norm is not None
                else np.zeros(nobs, dtype=bool)
            )
            if np.any(mask):
                a = pres[2 * ityp]
                ysim[:, mask] = f[None, mask] * np.exp(a * eps[:, mask])
    return ysim


def compute_npde(yobs, ysim):
    """
    Compute normalized prediction distribution errors (npde) via simulation.
    """
    nsim, nobs = ysim.shape
    npde = np.zeros(nobs)
    for i in range(nobs):
        ranks = rankdata(np.append(ysim[:, i], yobs[i]), method="average")
        u = (ranks[-1] - 0.5) / (nsim + 1)
        npde[i] = norm.ppf(u)
    return npde


def npde_tests(npde):
    """
    Basic NPDE tests: mean=0 (t-test), variance=1 (chi-square), normality (Shapiro).
    Returns a dict with test statistics and p-values.
    """
    npde = np.asarray(npde)
    npde = npde[np.isfinite(npde)]
    n = len(npde)
    if n < 3:
        return {"n": n, "mean_p": np.nan, "var_p": np.nan, "norm_p": np.nan}
    mean_stat, mean_p = ttest_1samp(npde, 0.0, nan_policy="omit")
    var = np.var(npde, ddof=1)
    var_stat = (n - 1) * var
    var_p = 2 * min(chi2.cdf(var_stat, df=n - 1), 1 - chi2.cdf(var_stat, df=n - 1))
    try:
        _, norm_p = shapiro(npde if n <= 5000 else npde[:5000])
    except Exception:
        norm_p = np.nan
    return {"n": n, "mean_p": mean_p, "var_p": var_p, "norm_p": norm_p}


def plot_npde(saemix_object, nsim=1000, pred_type="ppred", seed=None, show_tests=True):
    """
    Plot NPDE histogram, QQ plot, and two scatter diagnostics (vs predictor and vs pred).
    """
    plt = _require_matplotlib()
    data = saemix_object.data
    yobs = data.data[data.name_response].values
    ysim = simulate_observations(
        saemix_object, nsim=nsim, pred_type=pred_type, seed=seed
    )
    npde = compute_npde(yobs, ysim)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.hist(npde, bins=30, density=True, alpha=0.7)
    x = np.linspace(-4, 4, 200)
    ax.plot(x, norm.pdf(x), color="black")
    ax.set_title("NPDE histogram")
    ax.set_xlabel("NPDE")
    ax.set_ylabel("Density")
    qq = np.sort(npde)
    theo = norm.ppf((np.arange(1, len(qq) + 1) - 0.5) / len(qq))
    ax = axes[0, 1]
    ax.scatter(theo, qq, s=10)
    ax.plot(
        [theo.min(), theo.max()],
        [theo.min(), theo.max()],
        color="black",
        linestyle="--",
    )
    ax.set_title("NPDE QQ plot")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    xname = data.name_X if data.name_X else data.name_predictors[0]
    xpred = data.data[xname].values
    ax = axes[1, 0]
    ax.scatter(xpred, npde, s=10)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"NPDE vs {xname}")
    ax.set_xlabel(xname)
    ax.set_ylabel("NPDE")
    pred = saemix_object.predict(type=pred_type)
    ax = axes[1, 1]
    ax.scatter(pred, npde, s=10)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"NPDE vs {pred_type}")
    ax.set_xlabel(pred_type)
    ax.set_ylabel("NPDE")
    if show_tests:
        stats = npde_tests(npde)
        fig.suptitle(
            f"NPDE tests: mean p={stats['mean_p']:.3g}, var p={stats['var_p']:.3g}, normal p={stats['norm_p']:.3g}",
            y=0.98,
            fontsize=10,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_vpc(
    saemix_object,
    nsim=1000,
    pred_type="ppred",
    bins=10,
    ci=0.9,
    seed=None,
    by=None,
    ncols=2,
):
    """
    Visual predictive check with simulated quantile bands.
    If by is provided, facets are created per covariate level/bin.
    """
    plt = _require_matplotlib()
    data = saemix_object.data
    xname = data.name_X if data.name_X else data.name_predictors[0]
    x = data.data[xname].values
    y = data.data[data.name_response].values
    ysim = simulate_observations(
        saemix_object, nsim=nsim, pred_type=pred_type, seed=seed
    )
    q = np.array([0.05, 0.5, 0.95])
    groups = _get_by_groups(data, by, max_levels=6, nbins=4)
    n_panels = len(groups)
    ncols = min(max(1, ncols), n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(8 * ncols, 4 * nrows), squeeze=False
    )
    for gi, (label, gmask) in enumerate(groups):
        ax = axes[gi // ncols][gi % ncols]
        xg = x[gmask]
        yg = y[gmask]
        ysim_g = ysim[:, gmask]
        edges = np.quantile(xg, np.linspace(0, 1, bins + 1))
        mids = 0.5 * (edges[:-1] + edges[1:])
        obs_q = np.full((bins, len(q)), np.nan)
        sim_q = np.full((nsim, bins, len(q)), np.nan)
        for b in range(bins):
            mask = (xg >= edges[b]) & (
                xg <= edges[b + 1] if b == bins - 1 else xg < edges[b + 1]
            )
            if not np.any(mask):
                continue
            obs_q[b, :] = np.quantile(yg[mask], q)
            for s in range(nsim):
                sim_q[s, b, :] = np.quantile(ysim_g[s, mask], q)
        lo = (1 - ci) / 2
        hi = 1 - lo
        sim_med = np.nanquantile(sim_q, 0.5, axis=0)
        sim_lo = np.nanquantile(sim_q, lo, axis=0)
        sim_hi = np.nanquantile(sim_q, hi, axis=0)
        for i, label_q in enumerate(["5%", "50%", "95%"]):
            ax.plot(
                mids,
                obs_q[:, i],
                color="black",
                linestyle="-",
                label=f"Observed {label_q}" if i == 1 else None,
            )
            ax.plot(
                mids,
                sim_med[:, i],
                color="tab:blue",
                linestyle="--",
                label=f"Sim median {label_q}" if i == 1 else None,
            )
            ax.fill_between(
                mids, sim_lo[:, i], sim_hi[:, i], color="tab:blue", alpha=0.2
            )
        ax.set_title(f"VPC {label}")
        ax.set_xlabel(xname)
        ax.set_ylabel(data.name_response)
    for gi in range(n_panels, nrows * ncols):
        axes[gi // ncols][gi % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    return fig


def _normalize_ytype(ytype, ntypes):
    if ytype is None:
        return None
    ytype_arr = np.asarray(ytype).astype(int)
    if ytype_arr.size == 0:
        return ytype_arr
    if ntypes > 1 and ytype_arr.min() >= 1 and ytype_arr.max() == ntypes:
        return ytype_arr - 1
    return ytype_arr


def _get_by_groups(data, by, max_levels=6, nbins=4):
    if by is None:
        return [("All", np.ones(len(data.data), dtype=bool))]
    if by not in data.data.columns:
        return [("All", np.ones(len(data.data), dtype=bool))]
    col = data.data[by]
    if col.dtype.kind in {"O", "b"} or col.nunique() <= max_levels:
        groups = []
        for val in col.dropna().unique():
            groups.append((f"{by}={val}", col == val))
        return groups if groups else [("All", np.ones(len(data.data), dtype=bool))]
    # numeric: bin by quantiles
    edges = np.quantile(col.values, np.linspace(0, 1, nbins + 1))
    groups = []
    for b in range(nbins):
        mask = (col >= edges[b]) & (
            col <= edges[b + 1] if b == nbins - 1 else col < edges[b + 1]
        )
        groups.append((f"{by} bin {b+1}", mask))
    return groups


# =============================================================================
# Convergence Diagnostic Plots (Requirements 6.1, 6.2)
# =============================================================================


def plot_convergence(
    saemix_object,
    parameters=None,
    figsize=None,
    show_burnin=True,
    burnin_color="lightgray",
):
    """
    Plot parameter estimates versus iteration number for convergence diagnostics.

    Parameters
    ----------
    saemix_object : SaemixObject
        Fitted SAEM object with iteration history
    parameters : list of str, optional
        Parameter names to plot. If None, plots all parameters.
    figsize : tuple, optional
        Figure size (width, height). If None, uses global plot options.
    show_burnin : bool
        Whether to shade the burn-in region
    burnin_color : str
        Color for burn-in region shading

    Returns
    -------
    matplotlib.figure.Figure
        The convergence plot figure

    Notes
    -----
    Requires that the SAEM algorithm was run with iteration history recording
    (parpop attribute in results).
    """
    plt = _require_matplotlib()
    _apply_plot_options()
    figsize = _get_figsize(figsize) if figsize is None else figsize

    res = saemix_object.results
    model = saemix_object.model

    # Check if iteration history is available
    if res.parpop is None:
        raise ValueError(
            "Iteration history not available. "
            "Run SAEM with iteration history recording enabled."
        )

    parpop = res.parpop
    n_iter, n_params = parpop.shape

    # Get parameter names
    param_names = model.name_modpar if hasattr(model, "name_modpar") else None
    if param_names is None:
        param_names = [f"theta{i+1}" for i in range(n_params)]

    # Filter parameters if specified
    if parameters is not None:
        param_indices = []
        filtered_names = []
        for p in parameters:
            if isinstance(p, int):
                if 0 <= p < n_params:
                    param_indices.append(p)
                    filtered_names.append(param_names[p])
            elif p in param_names:
                param_indices.append(param_names.index(p))
                filtered_names.append(p)
        if not param_indices:
            raise ValueError(f"No valid parameters found. Available: {param_names}")
        param_names = filtered_names
    else:
        param_indices = list(range(n_params))

    n_plot = len(param_indices)
    ncols = min(3, n_plot)
    nrows = int(np.ceil(n_plot / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    iterations = np.arange(1, n_iter + 1)

    # Get burn-in iteration count from options if available
    burnin_iter = 0
    if hasattr(saemix_object, "options") and saemix_object.options:
        burnin_iter = saemix_object.options.get("nbiter_burn", 0)
        if burnin_iter == 0:
            # Try alternative key
            burnin_iter = saemix_object.options.get("nbiter", {})
            if isinstance(burnin_iter, dict):
                burnin_iter = burnin_iter.get("burn", 0)

    for idx, (param_idx, name) in enumerate(zip(param_indices, param_names)):
        ax = axes[idx // ncols][idx % ncols]

        # Plot parameter trajectory
        ax.plot(iterations, parpop[:, param_idx], color="tab:blue", linewidth=1)

        # Shade burn-in region
        if show_burnin and burnin_iter > 0:
            ax.axvspan(0, burnin_iter, alpha=0.3, color=burnin_color, label="Burn-in")
            ax.axvline(burnin_iter, color="gray", linestyle="--", linewidth=0.8)

        # Add final estimate line
        if res.fixed_effects is not None and param_idx < len(res.fixed_effects):
            ax.axhline(
                res.fixed_effects[param_idx],
                color="red",
                linestyle="--",
                linewidth=1,
                label="Final estimate",
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel(name)
        ax.set_title(f"Convergence: {name}")

    # Hide unused subplots
    for idx in range(n_plot, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    # Add legend to first subplot
    if show_burnin and burnin_iter > 0:
        axes[0][0].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


def plot_likelihood(
    saemix_object,
    figsize=None,
    show_burnin=True,
    burnin_color="lightgray",
):
    """
    Plot log-likelihood trajectory during SAEM estimation.

    Parameters
    ----------
    saemix_object : SaemixObject
        Fitted SAEM object with iteration history
    figsize : tuple, optional
        Figure size (width, height). If None, uses global plot options.
    show_burnin : bool
        Whether to shade the burn-in region
    burnin_color : str
        Color for burn-in region shading

    Returns
    -------
    matplotlib.figure.Figure
        The likelihood trajectory plot

    Notes
    -----
    Requires that the SAEM algorithm was run with likelihood tracking.
    If per-iteration likelihood is not available, displays the final likelihood.
    """
    plt = _require_matplotlib()
    _apply_plot_options()
    figsize = _get_figsize(figsize) if figsize is None else figsize

    res = saemix_object.results

    # Check if we have iteration-level likelihood
    ll_history = None
    if hasattr(res, "ll_history") and res.ll_history is not None:
        ll_history = res.ll_history
    elif res.allpar is not None:
        # Try to extract from allpar if it contains likelihood
        pass

    fig, ax = plt.subplots(figsize=figsize)

    # Get burn-in iteration count
    burnin_iter = 0
    if hasattr(saemix_object, "options") and saemix_object.options:
        burnin_iter = saemix_object.options.get("nbiter_burn", 0)
        if burnin_iter == 0:
            burnin_iter = saemix_object.options.get("nbiter", {})
            if isinstance(burnin_iter, dict):
                burnin_iter = burnin_iter.get("burn", 0)

    if ll_history is not None and len(ll_history) > 0:
        iterations = np.arange(1, len(ll_history) + 1)
        ax.plot(iterations, ll_history, color="tab:blue", linewidth=1.5)

        # Shade burn-in region
        if show_burnin and burnin_iter > 0:
            ax.axvspan(0, burnin_iter, alpha=0.3, color=burnin_color, label="Burn-in")
            ax.axvline(burnin_iter, color="gray", linestyle="--", linewidth=0.8)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log-likelihood")
        ax.set_title("Log-likelihood Trajectory")

    else:
        # No iteration history, show final likelihood as horizontal line
        if res.ll is not None:
            ax.axhline(
                res.ll, color="tab:blue", linewidth=2, label=f"Final LL: {res.ll:.2f}"
            )
            ax.set_ylabel("Log-likelihood")
            ax.set_title("Final Log-likelihood")
            ax.legend()
            # Add text annotation
            ax.text(
                0.5,
                0.5,
                f"Final Log-likelihood: {res.ll:.4f}\n\n"
                "(Per-iteration likelihood not recorded)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "Likelihood not available.\nRun likelihood computation first.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("Log-likelihood (Not Available)")

    fig.tight_layout()
    return fig


# =============================================================================
# Parameter-Covariate Relationship Plots (Requirements 6.3, 6.4)
# =============================================================================


def plot_parameters_vs_covariates(
    saemix_object,
    covariates=None,
    parameters=None,
    use_map=True,
    figsize=None,
    alpha=None,
):
    """
    Plot individual parameter estimates versus covariate values.

    Parameters
    ----------
    saemix_object : SaemixObject
        Fitted SAEM object
    covariates : list of str, optional
        Covariate names to plot. If None, uses all available covariates.
    parameters : list of str, optional
        Parameter names to plot. If None, plots all parameters.
    use_map : bool
        If True, use MAP estimates; otherwise use conditional mean estimates.
    figsize : tuple, optional
        Figure size (width, height). If None, uses global plot options.
    alpha : float, optional
        Transparency for scatter points. If None, uses global plot options.

    Returns
    -------
    matplotlib.figure.Figure
        The parameter vs covariate plot

    Notes
    -----
    For continuous covariates, displays scatter plots.
    For categorical covariates (<=6 unique values), displays box plots.
    """
    plt = _require_matplotlib()
    _apply_plot_options()
    figsize = _get_figsize(figsize)
    alpha = _get_alpha(alpha)

    data = saemix_object.data
    model = saemix_object.model
    res = saemix_object.results

    # Get individual parameter estimates
    if use_map:
        phi = res.map_phi if res.map_phi is not None else res.map_psi
    else:
        phi = res.cond_mean_phi if res.cond_mean_phi is not None else res.cond_mean_psi

    if phi is None:
        raise ValueError(
            "Individual parameter estimates not available. "
            "Run MAP estimation or conditional distribution estimation first."
        )

    # Convert to numpy array if DataFrame
    if hasattr(phi, "values"):
        phi = phi.values

    phi.shape[0]
    n_params = phi.shape[1]

    # Get parameter names
    param_names = model.name_modpar if hasattr(model, "name_modpar") else None
    if param_names is None:
        param_names = [f"theta{i+1}" for i in range(n_params)]

    # Get available covariates
    available_covs = data.name_covariates if data.name_covariates else []
    if not available_covs:
        raise ValueError("No covariates available in the data.")

    # Filter covariates
    if covariates is not None:
        cov_list = [c for c in covariates if c in available_covs]
        if not cov_list:
            raise ValueError(f"No valid covariates found. Available: {available_covs}")
    else:
        cov_list = available_covs

    # Filter parameters
    if parameters is not None:
        param_indices = []
        filtered_names = []
        for p in parameters:
            if isinstance(p, int):
                if 0 <= p < n_params:
                    param_indices.append(p)
                    filtered_names.append(param_names[p])
            elif p in param_names:
                param_indices.append(param_names.index(p))
                filtered_names.append(p)
        if not param_indices:
            raise ValueError(f"No valid parameters found. Available: {param_names}")
        param_names = filtered_names
    else:
        param_indices = list(range(n_params))

    # Get covariate values per subject (first observation per subject)
    df = data.data.copy()
    cov_per_subject = df.groupby("index")[cov_list].first()

    n_covs = len(cov_list)
    n_plot_params = len(param_indices)

    fig, axes = plt.subplots(n_plot_params, n_covs, figsize=figsize, squeeze=False)

    for i, (param_idx, pname) in enumerate(zip(param_indices, param_names)):
        for j, cov_name in enumerate(cov_list):
            ax = axes[i][j]
            cov_values = cov_per_subject[cov_name].values
            param_values = phi[:, param_idx]

            # Determine if categorical or continuous
            unique_vals = np.unique(cov_values[~np.isnan(cov_values)])
            is_categorical = len(unique_vals) <= 6

            if is_categorical:
                # Box plot for categorical
                groups = []
                labels = []
                for val in sorted(unique_vals):
                    mask = cov_values == val
                    if np.any(mask):
                        groups.append(param_values[mask])
                        labels.append(str(val))
                if groups:
                    bp = ax.boxplot(groups, labels=labels, patch_artist=True)
                    for patch in bp["boxes"]:
                        patch.set_facecolor("tab:blue")
                        patch.set_alpha(0.5)
            else:
                # Scatter plot for continuous
                valid_mask = ~np.isnan(cov_values)
                ax.scatter(
                    cov_values[valid_mask],
                    param_values[valid_mask],
                    alpha=alpha,
                    edgecolor="none",
                )
                # Add trend line
                if np.sum(valid_mask) > 2:
                    z = np.polyfit(cov_values[valid_mask], param_values[valid_mask], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(
                        np.min(cov_values[valid_mask]),
                        np.max(cov_values[valid_mask]),
                        100,
                    )
                    ax.plot(x_line, p(x_line), "r--", linewidth=1, alpha=0.7)

            ax.set_xlabel(cov_name)
            ax.set_ylabel(pname)
            if i == 0:
                ax.set_title(cov_name)

    fig.suptitle("Individual Parameters vs Covariates", y=1.02)
    fig.tight_layout()
    return fig


def plot_randeff_vs_covariates(
    saemix_object,
    covariates=None,
    use_map=True,
    figsize=None,
    alpha=None,
):
    """
    Plot random effects (eta) versus covariate values.

    Parameters
    ----------
    saemix_object : SaemixObject
        Fitted SAEM object
    covariates : list of str, optional
        Covariate names to plot. If None, uses all available covariates.
    use_map : bool
        If True, use MAP estimates; otherwise use conditional mean estimates.
    figsize : tuple, optional
        Figure size (width, height). If None, uses global plot options.
    alpha : float, optional
        Transparency for scatter points. If None, uses global plot options.

    Returns
    -------
    matplotlib.figure.Figure
        The random effects vs covariate plot

    Notes
    -----
    For continuous covariates, displays scatter plots with trend lines.
    For categorical covariates (<=6 unique values), displays box plots.
    """
    plt = _require_matplotlib()
    _apply_plot_options()
    figsize = _get_figsize(figsize)
    alpha = _get_alpha(alpha)

    data = saemix_object.data
    model = saemix_object.model
    res = saemix_object.results

    # Get random effects
    if use_map:
        eta = res.map_eta
    else:
        eta = res.cond_mean_eta

    if eta is None:
        raise ValueError(
            "Random effects not available. "
            "Run MAP estimation or conditional distribution estimation first."
        )

    if hasattr(eta, "values"):
        eta = eta.values

    eta.shape[0]
    n_eta = eta.shape[1]

    # Get parameter names for eta
    param_names = model.name_modpar if hasattr(model, "name_modpar") else None
    if param_names is None:
        eta_names = [f"eta{i+1}" for i in range(n_eta)]
    else:
        # Only include parameters with random effects
        indx_omega = model.indx_omega if hasattr(model, "indx_omega") else range(n_eta)
        eta_names = [f"eta_{param_names[i]}" for i in indx_omega[:n_eta]]

    # Get available covariates
    available_covs = data.name_covariates if data.name_covariates else []
    if not available_covs:
        raise ValueError("No covariates available in the data.")

    # Filter covariates
    if covariates is not None:
        cov_list = [c for c in covariates if c in available_covs]
        if not cov_list:
            raise ValueError(f"No valid covariates found. Available: {available_covs}")
    else:
        cov_list = available_covs

    # Get covariate values per subject
    df = data.data.copy()
    cov_per_subject = df.groupby("index")[cov_list].first()

    n_covs = len(cov_list)

    fig, axes = plt.subplots(n_eta, n_covs, figsize=figsize, squeeze=False)

    for i in range(n_eta):
        for j, cov_name in enumerate(cov_list):
            ax = axes[i][j]
            cov_values = cov_per_subject[cov_name].values
            eta_values = eta[:, i]

            # Determine if categorical or continuous
            unique_vals = np.unique(cov_values[~np.isnan(cov_values)])
            is_categorical = len(unique_vals) <= 6

            if is_categorical:
                # Box plot for categorical
                groups = []
                labels = []
                for val in sorted(unique_vals):
                    mask = cov_values == val
                    if np.any(mask):
                        groups.append(eta_values[mask])
                        labels.append(str(val))
                if groups:
                    bp = ax.boxplot(groups, labels=labels, patch_artist=True)
                    for patch in bp["boxes"]:
                        patch.set_facecolor("tab:green")
                        patch.set_alpha(0.5)
            else:
                # Scatter plot for continuous
                valid_mask = ~np.isnan(cov_values)
                ax.scatter(
                    cov_values[valid_mask],
                    eta_values[valid_mask],
                    alpha=alpha,
                    edgecolor="none",
                    color="tab:green",
                )
                # Add trend line
                if np.sum(valid_mask) > 2:
                    z = np.polyfit(cov_values[valid_mask], eta_values[valid_mask], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(
                        np.min(cov_values[valid_mask]),
                        np.max(cov_values[valid_mask]),
                        100,
                    )
                    ax.plot(x_line, p(x_line), "r--", linewidth=1, alpha=0.7)

            # Add reference line at 0
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

            ax.set_xlabel(cov_name)
            ax.set_ylabel(eta_names[i])
            if i == 0:
                ax.set_title(cov_name)

    fig.suptitle("Random Effects vs Covariates", y=1.02)
    fig.tight_layout()
    return fig


# =============================================================================
# Parameter Distribution Plots (Requirements 6.5, 6.6)
# =============================================================================


def plot_marginal_distribution(
    saemix_object,
    parameters=None,
    use_map=True,
    figsize=None,
    bins=30,
    show_density=True,
):
    """
    Plot marginal distributions of individual parameter estimates.

    Parameters
    ----------
    saemix_object : SaemixObject
        Fitted SAEM object
    parameters : list of str, optional
        Parameter names to plot. If None, plots all parameters.
    use_map : bool
        If True, use MAP estimates; otherwise use conditional mean estimates.
    figsize : tuple, optional
        Figure size (width, height). If None, uses global plot options.
    bins : int
        Number of histogram bins
    show_density : bool
        If True, overlay kernel density estimate

    Returns
    -------
    matplotlib.figure.Figure
        The marginal distribution plot

    Notes
    -----
    Displays histograms with optional kernel density estimates for each parameter.
    """
    plt = _require_matplotlib()
    _apply_plot_options()
    figsize = _get_figsize(figsize)
    from scipy.stats import gaussian_kde

    model = saemix_object.model
    res = saemix_object.results

    # Get individual parameter estimates
    if use_map:
        phi = res.map_phi if res.map_phi is not None else res.map_psi
    else:
        phi = res.cond_mean_phi if res.cond_mean_phi is not None else res.cond_mean_psi

    if phi is None:
        raise ValueError(
            "Individual parameter estimates not available. "
            "Run MAP estimation or conditional distribution estimation first."
        )

    # Convert to numpy array if DataFrame
    if hasattr(phi, "values"):
        phi = phi.values

    n_params = phi.shape[1]

    # Get parameter names
    param_names = model.name_modpar if hasattr(model, "name_modpar") else None
    if param_names is None:
        param_names = [f"theta{i+1}" for i in range(n_params)]

    # Filter parameters
    if parameters is not None:
        param_indices = []
        filtered_names = []
        for p in parameters:
            if isinstance(p, int):
                if 0 <= p < n_params:
                    param_indices.append(p)
                    filtered_names.append(param_names[p])
            elif p in param_names:
                param_indices.append(param_names.index(p))
                filtered_names.append(p)
        if not param_indices:
            raise ValueError(f"No valid parameters found. Available: {param_names}")
        param_names = filtered_names
    else:
        param_indices = list(range(n_params))

    n_plot = len(param_indices)
    ncols = min(3, n_plot)
    nrows = int(np.ceil(n_plot / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, (param_idx, name) in enumerate(zip(param_indices, param_names)):
        ax = axes[idx // ncols][idx % ncols]
        values = phi[:, param_idx]

        # Plot histogram
        ax.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.7,
            color="tab:blue",
            edgecolor="white",
        )

        # Overlay density estimate
        if show_density and len(values) > 3:
            try:
                kde = gaussian_kde(values)
                x_range = np.linspace(np.min(values), np.max(values), 200)
                ax.plot(x_range, kde(x_range), color="red", linewidth=2, label="KDE")
            except Exception:
                pass  # Skip KDE if it fails

        # Add population mean line
        if res.fixed_effects is not None and param_idx < len(res.fixed_effects):
            ax.axvline(
                res.fixed_effects[param_idx],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=f"Pop. mean: {res.fixed_effects[param_idx]:.3g}",
            )

        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution: {name}")
        ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(n_plot, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Marginal Parameter Distributions", y=1.02)
    fig.tight_layout()
    return fig


def plot_correlations(
    saemix_object,
    use_map=True,
    figsize=None,
    cmap="RdBu_r",
    annot=True,
):
    """
    Plot correlation matrix of random effects.

    Parameters
    ----------
    saemix_object : SaemixObject
        Fitted SAEM object
    use_map : bool
        If True, use MAP estimates; otherwise use conditional mean estimates.
    figsize : tuple, optional
        Figure size (width, height). If None, uses global plot options.
    cmap : str
        Colormap for the correlation matrix
    annot : bool
        If True, annotate cells with correlation values

    Returns
    -------
    matplotlib.figure.Figure
        The correlation matrix plot

    Notes
    -----
    Displays a heatmap of correlations between random effects (eta).
    """
    plt = _require_matplotlib()
    _apply_plot_options()
    figsize = _get_figsize(figsize)

    model = saemix_object.model
    res = saemix_object.results

    # Get random effects
    if use_map:
        eta = res.map_eta
    else:
        eta = res.cond_mean_eta

    if eta is None:
        raise ValueError(
            "Random effects not available. "
            "Run MAP estimation or conditional distribution estimation first."
        )

    if hasattr(eta, "values"):
        eta = eta.values

    n_eta = eta.shape[1]

    if n_eta < 2:
        raise ValueError("At least 2 random effects are required for correlation plot.")

    # Get parameter names for eta
    param_names = model.name_modpar if hasattr(model, "name_modpar") else None
    if param_names is None:
        eta_names = [f"eta{i+1}" for i in range(n_eta)]
    else:
        indx_omega = model.indx_omega if hasattr(model, "indx_omega") else range(n_eta)
        eta_names = [f"eta_{param_names[i]}" for i in indx_omega[:n_eta]]

    # Compute correlation matrix
    corr_matrix = np.corrcoef(eta.T)

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation")

    # Set ticks and labels
    ax.set_xticks(np.arange(n_eta))
    ax.set_yticks(np.arange(n_eta))
    ax.set_xticklabels(eta_names, rotation=45, ha="right")
    ax.set_yticklabels(eta_names)

    # Annotate with correlation values
    if annot:
        for i in range(n_eta):
            for j in range(n_eta):
                text_color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{corr_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                )

    ax.set_title("Random Effects Correlation Matrix")
    fig.tight_layout()
    return fig
