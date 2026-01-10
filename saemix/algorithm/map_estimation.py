from typing import Callable

import numpy as np
from scipy.optimize import minimize

from saemix.utils import cutoff, transphi


def conditional_distribution_c(
    phi1, phii, idi, xi, yi, mphi, idx, iomega, trpar, model, pres, err, ytype=None
):
    """
    条件分布函数（用于structural模型），用于MAP估计

    参数
    -----
    phi1 : np.ndarray
        待优化的参数向量（只有随机效应部分）
    phii : np.ndarray
        完整参数向量
    idi : np.ndarray
        个体ID向量（全为1）
    xi : np.ndarray 或 pandas.DataFrame
        预测变量矩阵
    yi : np.ndarray
        观测值向量
    mphi : np.ndarray
        总体均值参数
    idx : np.ndarray
        随机效应的索引
    iomega : np.ndarray
        Omega的逆矩阵（仅随机效应部分）
    trpar : np.ndarray
        参数变换类型
    model : callable
        模型函数
    pres : np.ndarray
        误差模型参数
    err : list[str]
        误差模型类型列表
    ytype : np.ndarray, optional
        响应类型

    返回
    -----
    float
        条件分布的负对数（用于最小化）
    """
    phii_copy = phii.copy()
    phii_copy[idx] = phi1

    psii = transphi(phii_copy.reshape(1, -1), trpar)
    if psii.ndim == 1:
        psii = psii.reshape(1, -1)

    fi = model(psii, idi, xi)

    if ytype is not None:
        ind_exp = [i for i, e in enumerate(err) if e == "exponential"]
        for ityp in ind_exp:
            mask = (ytype == ityp) if hasattr(ytype, "__iter__") else (ytype == ityp)
            if np.any(mask):
                fi[mask] = np.log(cutoff(fi[mask]))

    gi = error_function(fi, pres, err, ytype)
    Uy = np.sum(0.5 * ((yi - fi) / gi) ** 2 + np.log(gi))

    dphi = phi1 - mphi
    Uphi = 0.5 * np.sum(dphi @ iomega * dphi)

    return Uy + Uphi


def conditional_distribution_d(
    phi1, phii, idi, xi, yi, mphi, idx, iomega, trpar, model
):
    """
    条件分布函数（用于likelihood模型），用于MAP估计

    参数
    -----
    phi1 : np.ndarray
        待优化的参数向量（只有随机效应部分）
    phii : np.ndarray
        完整参数向量
    idi : np.ndarray
        个体ID向量（全为1）
    xi : np.ndarray 或 pandas.DataFrame
        预测变量矩阵
    yi : np.ndarray
        观测值向量（用于likelihood模型）
    mphi : np.ndarray
        总体均值参数
    idx : np.ndarray
        随机效应的索引
    iomega : np.ndarray
        Omega的逆矩阵（仅随机效应部分）
    trpar : np.ndarray
        参数变换类型
    model : callable
        模型函数（返回对数似然）

    返回
    -----
    float
        条件分布的负对数（用于最小化）
    """
    phii_copy = phii.copy()
    phii_copy[idx] = phi1

    psii = transphi(phii_copy.reshape(1, -1), trpar)
    if psii.ndim == 1:
        psii = psii.reshape(1, -1)

    fi = model(psii, idi, xi)
    Uy = -np.sum(fi)

    dphi = phi1 - mphi
    Uphi = 0.5 * np.sum(dphi @ iomega * dphi)

    return Uy + Uphi


def error_function(f, pres, err, ytype=None):
    """
    计算误差模型的标准差

    参数
    -----
    f : np.ndarray
        预测值
    pres : np.ndarray
        误差模型参数 [a, b]
    err : list[str]
        误差模型类型列表
    ytype : np.ndarray, optional
        响应类型

    返回
    -----
    np.ndarray
        标准差
    """
    f = np.array(f)
    g = f.copy()

    if ytype is not None and len(np.unique(ytype)) > 1:
        ytype_arr = np.asarray(ytype).astype(int)
        if len(err) > 1 and ytype_arr.min() >= 1 and ytype_arr.max() == len(err):
            ytype_arr = ytype_arr - 1
        for ityp in np.unique(ytype_arr):
            mask = ytype_arr == ityp
            idx = int(ityp)
            if idx < len(err):
                ab = pres[2 * idx : 2 * idx + 2]
                g[mask] = error_type(f[mask], ab, err[idx])
    else:
        if isinstance(err, list):
            err_type = err[0] if len(err) > 0 else "constant"
        else:
            err_type = err
        ab = pres[:2]
        g = error_type(f, ab, err_type)

    return cutoff(g)


def error_type(f, ab, err_type):
    """
    计算特定类型的误差模型标准差

    参数
    -----
    f : np.ndarray
        预测值
    ab : np.ndarray
        误差参数 [a, b]
    err_type : str
        误差模型类型

    返回
    -----
    np.ndarray
        标准差
    """
    f = np.array(f)
    ab = np.array(ab)

    # Align with R saemix: always use sqrt(a^2 + b^2*f^2)
    # This works for all error models (constant: b=0, proportional: a=0, combined: both)
    g = np.sqrt(ab[0] ** 2 + (ab[1] * f) ** 2)
    return cutoff(g)


def map_saemix(saemix_object):
    """
    计算个体参数的MAP估计

    参数
    -----
    saemix_object : SaemixObject
        SAEM拟合结果对象

    返回
    -----
    SaemixObject
        更新了map_phi和map_psi的结果对象
    """
    from saemix.data import SaemixData
    from saemix.model import SaemixModel
    from saemix.results import SaemixObject

    if not isinstance(saemix_object, SaemixObject):
        raise TypeError("saemix_object must be a SaemixObject instance")

    model = saemix_object.model
    data = saemix_object.data
    results = saemix_object.results

    if results.omega is None:
        raise ValueError("Omega matrix not found. Run SAEM algorithm first.")

    if results.mean_phi is None:
        raise ValueError("mean_phi not found. Run SAEM algorithm first.")

    if results.respar is None:
        results.respar = model.error_init

    i1_omega2 = model.indx_omega
    omega_sub = results.omega[np.ix_(i1_omega2, i1_omega2)]

    try:
        iomega_phi1 = np.linalg.inv(omega_sub)
    except np.linalg.LinAlgError:
        iomega_phi1 = np.eye(len(i1_omega2))

    id_col = data.data[data.name_group].values
    xind = data.data[data.name_predictors].values
    yobs = data.data[data.name_response].values

    id_list = np.unique(id_col)
    N = len(id_list)

    phi_map = results.mean_phi.copy()

    if saemix_object.options.get("warnings", True):
        print("Estimating the individual parameters, please wait a few moments...")

    for i in range(N):
        if saemix_object.options.get("warnings", True) and (i + 1) % 10 == 0:
            print(".", end="", flush=True)

        isuj = id_list[i]
        mask = id_col == isuj
        xi = xind[mask]
        yi = yobs[mask]

        idi = np.zeros(len(yi), dtype=int)

        mean_phi1 = results.mean_phi[i, i1_omega2]
        phii = results.mean_phi[i, :].copy()
        phi1 = phii[i1_omega2]

        if model.modeltype == "structural":
            ytype = None
            if hasattr(data, "name_ytype") and data.name_ytype:
                if data.name_ytype in data.data.columns:
                    ytype = data.data[mask][data.name_ytype].values
                elif "ytype" in data.data.columns:
                    ytype = data.data[mask]["ytype"].values

            def objective(phi1_opt):
                return conditional_distribution_c(
                    phi1_opt,
                    phii,
                    idi,
                    xi,
                    yi,
                    mean_phi1,
                    i1_omega2,
                    iomega_phi1,
                    model.transform_par,
                    model.model,
                    results.respar,
                    model.error_model,
                    ytype,
                )

        else:

            def objective(phi1_opt):
                return conditional_distribution_d(
                    phi1_opt,
                    phii,
                    idi,
                    xi,
                    yi,
                    mean_phi1,
                    i1_omega2,
                    iomega_phi1,
                    model.transform_par,
                    model.model,
                )

        try:
            res = minimize(objective, phi1, method="BFGS")
            phi_map[i, i1_omega2] = res.x
        except Exception as e:
            if saemix_object.options.get("warnings", True):
                print(f"\nWarning: MAP estimation failed for subject {isuj}: {e}")
            phi_map[i, i1_omega2] = phi1

    if saemix_object.options.get("warnings", True):
        print()

    from saemix.utils import transphi

    map_psi = transphi(phi_map, model.transform_par)
    map_eta = None
    if results.mean_phi is not None:
        try:
            mean_phi = results.mean_phi
            if mean_phi.ndim == 1:
                mean_phi = mean_phi.reshape(1, -1)
            map_eta = phi_map - mean_phi
        except Exception:
            map_eta = None

    results.map_phi = phi_map
    results.map_psi = map_psi
    results.map_eta = map_eta

    if model.name_modpar is not None and len(model.name_modpar) > 0:
        import pandas as pd

        map_psi_df = pd.DataFrame(map_psi, columns=model.name_modpar)
        map_phi_df = pd.DataFrame(phi_map)
        results.map_psi = map_psi_df
        results.map_phi = map_phi_df
        if map_eta is not None:
            results.map_eta = pd.DataFrame(map_eta)

    return saemix_object
