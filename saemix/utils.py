import warnings
import numpy as np
from scipy.stats import norm

# 数值常量
LOG_EPS = 1e-10
LOGIT_EPS = 1e-10


def transphi(phi, tr, verbose: bool = False):
    """
    将 phi (未变换的参数) 转换为 psi (变换后的参数)

    增加数值防护，防止 NaN/Inf

    参数
    -----
    phi : np.ndarray
        未变换的参数矩阵
    tr : np.ndarray
        变换类型向量 (0=normal, 1=log-normal, 2=probit, 3=logit)
    verbose : bool, optional
        是否输出警告信息，默认 False

    返回
    -----
    np.ndarray
        变换后的参数矩阵

    Raises
    ------
    ValueError
        如果变换产生溢出（Inf 值）
    """
    psi = phi.copy()
    if phi.ndim == 1:
        psi = psi.reshape(1, -1)
        was_1d = True
    else:
        was_1d = False

    tr = np.array(tr)

    # Log 变换 (tr == 1): psi = exp(phi)
    i1 = np.where(tr == 1)[0]
    if len(i1) > 0:
        # 无需 clip，因为 exp 对任何有限输入都产生有限输出
        psi[:, i1] = np.exp(psi[:, i1])
        # 检查溢出
        if np.any(np.isinf(psi[:, i1])):
            max_input = np.max(phi[:, i1]) if phi.ndim > 1 else np.max(phi[i1])
            raise ValueError(
                f"Log transformation overflow: input values too large. "
                f"Max input: {max_input}"
            )

    # Probit 变换 (tr == 2): psi = norm.cdf(phi)
    i2 = np.where(tr == 2)[0]
    if len(i2) > 0:
        psi[:, i2] = norm.cdf(psi[:, i2])

    # Logit 变换 (tr == 3): psi = 1 / (1 + exp(-phi))
    i3 = np.where(tr == 3)[0]
    if len(i3) > 0:
        psi[:, i3] = 1 / (1 + np.exp(-psi[:, i3]))

    if was_1d:
        psi = psi.flatten()

    return psi


def transpsi(psi, tr, verbose: bool = False):
    """
    将 psi (变换后的参数) 转换为 phi (未变换的参数)

    增加数值防护，防止 NaN/Inf

    参数
    -----
    psi : np.ndarray
        变换后的参数矩阵
    tr : np.ndarray
        变换类型向量 (0=normal, 1=log-normal, 2=probit, 3=logit)
    verbose : bool, optional
        是否输出警告信息，默认 False

    返回
    -----
    np.ndarray
        未变换的参数矩阵

    Raises
    ------
    ValueError
        如果变换产生非有限值（NaN 或 Inf）
    """
    phi = psi.copy()
    if phi.ndim == 1:
        phi = phi.reshape(1, -1)
        was_1d = True
    else:
        was_1d = False

    tr = np.array(tr)

    # Log 逆变换 (tr == 1): phi = log(psi)
    i1 = np.where(tr == 1)[0]
    if len(i1) > 0:
        # Clip 防止 log(0) = -Inf
        original_values = phi[:, i1].copy()
        clipped = np.clip(phi[:, i1], LOG_EPS, None)
        if verbose and np.any(original_values < LOG_EPS):
            n_clipped = np.sum(original_values < LOG_EPS)
            warnings.warn(
                f"Log inverse transform: {n_clipped} values clipped to {LOG_EPS}"
            )
        phi[:, i1] = np.log(clipped)

    # Probit 逆变换 (tr == 2): phi = norm.ppf(psi)
    i2 = np.where(tr == 2)[0]
    if len(i2) > 0:
        original_values = phi[:, i2].copy()
        clipped = np.clip(phi[:, i2], LOGIT_EPS, 1 - LOGIT_EPS)
        if verbose and np.any(
            (original_values < LOGIT_EPS) | (original_values > 1 - LOGIT_EPS)
        ):
            warnings.warn("Probit inverse transform: values clipped to (eps, 1-eps)")
        phi[:, i2] = norm.ppf(clipped)

    # Logit 逆变换 (tr == 3): phi = log(psi / (1 - psi))
    i3 = np.where(tr == 3)[0]
    if len(i3) > 0:
        # Clip 防止 log(0) 和 log(inf)
        original_values = phi[:, i3].copy()
        clipped = np.clip(phi[:, i3], LOGIT_EPS, 1 - LOGIT_EPS)
        if verbose and np.any(
            (original_values < LOGIT_EPS) | (original_values > 1 - LOGIT_EPS)
        ):
            warnings.warn(
                f"Logit inverse transform: values clipped to ({LOGIT_EPS}, {1-LOGIT_EPS})"
            )
        phi[:, i3] = np.log(clipped / (1 - clipped))

    # 最终有限性检查
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


def cutoff(x, eps=1e-10):
    """
    将小于 eps 的值截断为 eps
    """
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
