import warnings

import numpy as np
from scipy.stats import norm

# Numerical constants for parameter transformations and numerical safety
LOG_EPS = 1e-10
LOGIT_EPS = 1e-10


def _apply_log_transform(psi_vals: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Apply log transformation: psi = exp(phi)

    The log transform maps untransformed parameters to positive values via:
        psi = exp(phi)

    This does not need clipping since exp produces finite output for all finite inputs.

    Parameters
    ----------
    psi_vals : np.ndarray
        Untransformed parameter values to apply log transform to.
    verbose : bool, optional
        Whether to output warning messages (default False).

    Returns
    -------
    np.ndarray
        Log-transformed parameter values.

    Raises
    ------
    ValueError
        If log transformation produces Inf (input values too large).
    """
    transformed = np.exp(psi_vals)

    # Check for overflow
    if np.any(np.isinf(transformed)):
        max_input = np.max(psi_vals)
        raise ValueError(
            f"Log transformation overflow: input values too large. "
            f"Max input: {max_input}"
        )

    return transformed


def _apply_probit_transform(psi_vals: np.ndarray) -> np.ndarray:
    """
    Apply probit transformation: psi = norm.cdf(phi)

    The probit transform maps untransformed parameters to [0, 1] via:
        psi = Phi(phi)

    where Phi is the standard normal cumulative distribution function.

    Parameters
    ----------
    psi_vals : np.ndarray
        Untransformed parameter values to apply probit transform to.

    Returns
    -------
    np.ndarray
        Probit-transformed parameter values in [0, 1].
    """
    return norm.cdf(psi_vals)


def _apply_logit_transform(psi_vals: np.ndarray) -> np.ndarray:
    """
    Apply logistic transformation: psi = 1 / (1 + exp(-phi))

    The logistic transform maps untransformed parameters to [0, 1] via:
        psi = 1 / (1 + exp(-phi))

    Parameters
    ----------
    psi_vals : np.ndarray
        Untransformed parameter values to apply logistic transform to.

    Returns
    -------
    np.ndarray
        Logistic-transformed parameter values in [0, 1].
    """
    # Formula: psi = 1 / (1 + exp(-phi)) = exp(phi) / (1 + exp(phi))
    # For numerical stability: psi = 1 - 1/(1 + exp(-phi)) + exp(-phi) / (1 + exp(-phi))
    exp_phi = np.exp(-psi_vals)
    transformed = 1 / (1 + exp_phi)

    return transformed


def _inverse_log_transform(psi_vals: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Apply inverse log transformation: phi = log(psi)

    Clips input to prevent log(0) = -Inf.

    Parameters
    ----------
    psi_vals : np.ndarray
        Log-transformed parameter values to invert.
    verbose : bool, optional
        Whether to output warning messages (default False).

    Returns
    -------
    np.ndarray
        Untransformed parameter values (original scale).
    """
    original_values = psi_vals.copy()
    clipped = np.clip(psi_vals, LOG_EPS, None)

    if verbose and np.any(original_values < LOG_EPS):
        n_clipped = np.sum(original_values < LOG_EPS)
        warnings.warn(f"Log inverse transform: {n_clipped} values clipped to {LOG_EPS}")

    return np.log(clipped)


def _inverse_probit_transform(
    psi_vals: np.ndarray, verbose: bool = False
) -> np.ndarray:
    """
    Apply inverse probit transformation: phi = norm.ppf(psi)

    Clips input to [LOGIT_EPS, 1 - LOGIT_EPS] before applying norm.ppf.

    Parameters
    ----------
    psi_vals : np.ndarray
        Probit-transformed parameter values to invert.
    verbose : bool, optional
        Whether to output warning messages (default False).

    Returns
    -------
    np.ndarray
        Untransformed parameter values (original scale).
    """
    original_values = psi_vals.copy()
    clipped = np.clip(psi_vals, LOGIT_EPS, 1 - LOGIT_EPS)

    if verbose and np.any(
        (original_values < LOGIT_EPS) | (original_values > 1 - LOGIT_EPS)
    ):
        warnings.warn(
            f"Probit inverse transform: values clipped to ({LOGIT_EPS}, {1 - LOGIT_EPS})"
        )

    return norm.ppf(clipped)


def _inverse_logit_transform(psi_vals: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Apply inverse logit transformation: phi = log(psi / (1 - psi))

    Clips input to [LOGIT_EPS, 1 - LOGIT_EPS] to prevent log(0) and log(inf).

    Parameters
    ----------
    psi_vals : np.ndarray
        Logit-transformed parameter values to invert.
    verbose : bool, optional
        Whether to output warning messages (default False).

    Returns
    -------
    np.ndarray
        Untransformed parameter values (original scale).
    """
    original_values = psi_vals.copy()
    clipped = np.clip(psi_vals, LOGIT_EPS, 1 - LOGIT_EPS)

    if verbose and np.any(
        (original_values < LOGIT_EPS) | (original_values > 1 - LOGIT_EPS)
    ):
        warnings.warn(
            f"Logit inverse transform: values clipped to ({LOGIT_EPS}, {1 - LOGIT_EPS})"
        )

    return np.log(clipped / (1 - clipped))


def transphi(phi, tr, verbose: bool = False):
    """
    Transform phi (untransformed parameters) to psi (transformed parameters).

    Applies parameter transformations element-wise based on transform type vector:
    - 0: normal (identity)
    - 1: log-normal (exp transform)
    - 2: probit (normal CDF transform)
    - 3: logit (logistic transform)

    Includes numerical safety checks to prevent NaN/Inf.

    Parameters
    ----------
    phi : np.ndarray
        Untransformed parameter matrix.
    tr : np.ndarray
        Transform type vector (0=normal, 1=log-normal, 2=probit, 3=logit).
    verbose : bool, optional
        Whether to output warning messages (default False).

    Returns
    -------
    np.ndarray
        Transformed parameter matrix.

    Raises
    ------
    ValueError
        If transformation produces overflow (Inf values).
    """
    psi = phi.copy()
    if phi.ndim == 1:
        psi = psi.reshape(1, -1)
        was_1d = True
    else:
        was_1d = False

    tr = np.array(tr)

    # Apply log transform (tr == 1): psi = exp(phi)
    i1 = np.where(tr == 1)[0]
    if len(i1) > 0:
        psi[:, i1] = _apply_log_transform(psi[:, i1], verbose)

    # Apply probit transform (tr == 2): psi = norm.cdf(phi)
    i2 = np.where(tr == 2)[0]
    if len(i2) > 0:
        psi[:, i2] = _apply_probit_transform(psi[:, i2])

    # Apply logit transform (tr == 3): psi = 1 / (1 + exp(-phi))
    i3 = np.where(tr == 3)[0]
    if len(i3) > 0:
        psi[:, i3] = _apply_logit_transform(psi[:, i3])

    if was_1d:
        psi = psi.flatten()

    # Final finite check
    if np.any(~np.isfinite(psi)):
        bad_indices = np.where(~np.isfinite(psi))
        raise ValueError(
            f"Transformation produced non-finite values at indices {bad_indices}. "
            f"Input range: [{np.nanmin(phi)}, {np.nanmax(phi)}]"
        )

    return psi


def transpsi(psi, tr, verbose: bool = False):
    """
    Transform psi (transformed parameters) to phi (untransformed parameters).

    Applies inverse parameter transformations element-wise based on transform type vector:
    - 0: normal (identity)
    - 1: log-normal (log transform)
    - 2: probit (normal PPF transform)
    - 3: logit (logistic inverse transform)

    Includes numerical safety checks with clipping to prevent log(0) = -Inf.

    Parameters
    ----------
    psi : np.ndarray
        Transformed parameter matrix.
    tr : np.ndarray
        Transform type vector (0=normal, 1=log-normal, 2=probit, 3=logit).
    verbose : bool, optional
        Whether to output warning messages (default False).

    Returns
    -------
    np.ndarray
        Untransformed parameter matrix.

    Raises
    ------
    ValueError
        If transformation produces non-finite values (NaN or Inf).
    """
    phi = psi.copy()
    if phi.ndim == 1:
        phi = phi.reshape(1, -1)
        was_1d = True
    else:
        was_1d = False

    tr = np.array(tr)

    # Apply inverse log transform (tr == 1): phi = log(psi)
    i1 = np.where(tr == 1)[0]
    if len(i1) > 0:
        phi[:, i1] = _inverse_log_transform(psi[:, i1], verbose)

    # Apply inverse probit transform (tr == 2): phi = norm.ppf(psi)
    i2 = np.where(tr == 2)[0]
    if len(i2) > 0:
        phi[:, i2] = _inverse_probit_transform(psi[:, i2], verbose)

    # Apply inverse logit transform (tr == 3): phi = log(psi / (1 - psi))
    i3 = np.where(tr == 3)[0]
    if len(i3) > 0:
        phi[:, i3] = _inverse_logit_transform(psi[:, i3], verbose)

    # Final finite check
    if np.any(~np.isfinite(phi)):
        bad_indices = np.where(~np.isfinite(phi))
        raise ValueError(
            f"Transformation produced non-finite values at indices {bad_indices}. "
            f"Input range: [{np.nanmin(psi)}, {np.nanmax(psi)}]"
        )

    if was_1d:
        phi = phi.flatten()

    return phi


def compute_phi(psi, transform_par):
    """
    从 psi 计算 phi（transpsi 的别名）
    """
    return transpsi(psi, transform_par)


def compute_psi(phi, transform_par):
    """
    从 phi 计算 psi（transphi 的别名）
    """
    return transphi(phi, transform_par)


def mydiag(nrow=None, ncol=None, x=None):
    """
    创建对角线矩阵，类似于 R 的 diag() 函数

    参数
    -----
    nrow : int, optional
        行数
    ncol : int, optional
        列数
    x : array-like, optional
        对角线元素

    返回
    -----
    np.ndarray
        对角线矩阵
    """
    if x is None:
        if nrow is None:
            return np.array([])
        if ncol is None:
            ncol = nrow
        return np.eye(nrow, ncol)
    else:
        x = np.array(x)
        if x.ndim == 0:
            if nrow is None:
                nrow = 1
            if ncol is None:
                ncol = 1
            return np.eye(nrow, ncol) * x
        else:
            return np.diag(x)


def cutoff(x, eps=None):
    """
    将小于 eps 的值截断为 eps

    参数
    -----
    x : array-like
        输入数组
    eps : float, optional
        截断阈值，默认为最小正浮点数 (np.finfo(float).tiny ≈ 2.225e-308)
        以匹配 R 的 .Machine$double.xmin
    """
    if eps is None:
        eps = np.finfo(float).tiny
    x = np.array(x)
    return np.maximum(x, eps)


def id_to_index(user_id: int) -> int:
    """
    将用户面向的 1-based ID 转换为内部 0-based 索引。

    Parameters
    ----------
    user_id : int
        用户提供的 1-based ID

    Returns
    -------
    int
        内部使用的 0-based 索引

    Raises
    ------
    ValueError
        如果 user_id < 1

    Examples
    --------
    >>> id_to_index(1)
    0
    >>> id_to_index(5)
    4
    """
    if user_id < 1:
        raise ValueError(f"User ID must be >= 1, got {user_id}")
    return user_id - 1


def index_to_id(index: int) -> int:
    """
    将内部 0-based 索引转换为用户面向的 1-based ID。

    Parameters
    ----------
    index : int
        内部使用的 0-based 索引

    Returns
    -------
    int
        用户面向的 1-based ID

    Raises
    ------
    ValueError
        如果 index < 0

    Examples
    --------
    >>> index_to_id(0)
    1
    >>> index_to_id(4)
    5
    """
    if index < 0:
        raise ValueError(f"Index must be >= 0, got {index}")
    return index + 1
