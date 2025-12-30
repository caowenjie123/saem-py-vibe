import numpy as np
from scipy.stats import norm


def transphi(phi, tr):
    """
    将 phi (未变换的参数) 转换为 psi (变换后的参数)
    
    参数
    -----
    phi : np.ndarray
        未变换的参数矩阵
    tr : np.ndarray
        变换类型向量 (0=normal, 1=log-normal, 2=probit, 3=logit)
    
    返回
    -----
    np.ndarray
        变换后的参数矩阵
    """
    psi = phi.copy()
    if phi.ndim == 1:
        psi = psi.reshape(1, -1)
        was_1d = True
    else:
        was_1d = False
    
    tr = np.array(tr)
    
    i1 = np.where(tr == 1)[0]
    if len(i1) > 0:
        psi[:, i1] = np.exp(psi[:, i1])
    
    i2 = np.where(tr == 2)[0]
    if len(i2) > 0:
        psi[:, i2] = norm.cdf(psi[:, i2])
    
    i3 = np.where(tr == 3)[0]
    if len(i3) > 0:
        psi[:, i3] = 1 / (1 + np.exp(-psi[:, i3]))
    
    if was_1d:
        psi = psi.flatten()
    
    return psi


def transpsi(psi, tr):
    """
    将 psi (变换后的参数) 转换为 phi (未变换的参数)
    
    参数
    -----
    psi : np.ndarray
        变换后的参数矩阵
    tr : np.ndarray
        变换类型向量 (0=normal, 1=log-normal, 2=probit, 3=logit)
    
    返回
    -----
    np.ndarray
        未变换的参数矩阵
    """
    phi = psi.copy()
    if phi.ndim == 1:
        phi = phi.reshape(1, -1)
        was_1d = True
    else:
        was_1d = False
    
    tr = np.array(tr)
    
    i1 = np.where(tr == 1)[0]
    if len(i1) > 0:
        phi[:, i1] = np.log(phi[:, i1])
    
    i2 = np.where(tr == 2)[0]
    if len(i2) > 0:
        phi[:, i2] = norm.ppf(np.clip(phi[:, i2], 1e-10, 1 - 1e-10))
    
    i3 = np.where(tr == 3)[0]
    if len(i3) > 0:
        phi[:, i3] = np.log(phi[:, i3] / (1 - phi[:, i3]))
    
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