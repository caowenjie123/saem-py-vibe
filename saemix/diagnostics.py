import numpy as np
from scipy.stats import norm, rankdata, ttest_1samp, chi2, shapiro
from saemix.algorithm.map_estimation import error_function


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting utilities") from exc
    return plt


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
    ytype = data.data['ytype'].values if 'ytype' in data.data.columns else None
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
    idx = data.data['index'].values
    subjects = np.unique(idx)
    n = min(n, len(subjects))
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
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
    ax.plot([theo.min(), theo.max()], [theo.min(), theo.max()], color="black", linestyle="--")
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
        raise ValueError("Eta not available; run MAP estimation or compute conditional means.")
    if hasattr(eta, "values"):
        eta = eta.values
    npar = eta.shape[1]
    ncols = int(np.ceil(np.sqrt(npar)))
    nrows = int(np.ceil(npar / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
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
    ytype = data.data['ytype'].values if 'ytype' in data.data.columns else None
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
            mask = ytype_norm == ityp if ytype_norm is not None else np.zeros(nobs, dtype=bool)
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
    ysim = simulate_observations(saemix_object, nsim=nsim, pred_type=pred_type, seed=seed)
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
    ax.plot([theo.min(), theo.max()], [theo.min(), theo.max()], color="black", linestyle="--")
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


def plot_vpc(saemix_object, nsim=1000, pred_type="ppred", bins=10, ci=0.9, seed=None, by=None, ncols=2):
    """
    Visual predictive check with simulated quantile bands.
    If by is provided, facets are created per covariate level/bin.
    """
    plt = _require_matplotlib()
    data = saemix_object.data
    xname = data.name_X if data.name_X else data.name_predictors[0]
    x = data.data[xname].values
    y = data.data[data.name_response].values
    ysim = simulate_observations(saemix_object, nsim=nsim, pred_type=pred_type, seed=seed)
    q = np.array([0.05, 0.5, 0.95])
    groups = _get_by_groups(data, by, max_levels=6, nbins=4)
    n_panels = len(groups)
    ncols = min(max(1, ncols), n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4 * nrows), squeeze=False)
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
            mask = (xg >= edges[b]) & (xg <= edges[b + 1] if b == bins - 1 else xg < edges[b + 1])
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
            ax.plot(mids, obs_q[:, i], color="black", linestyle="-", label=f"Observed {label_q}" if i == 1 else None)
            ax.plot(mids, sim_med[:, i], color="tab:blue", linestyle="--", label=f"Sim median {label_q}" if i == 1 else None)
            ax.fill_between(mids, sim_lo[:, i], sim_hi[:, i], color="tab:blue", alpha=0.2)
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
        mask = (col >= edges[b]) & (col <= edges[b + 1] if b == nbins - 1 else col < edges[b + 1])
        groups.append((f"{by} bin {b+1}", mask))
    return groups
