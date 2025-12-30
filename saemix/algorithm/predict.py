import numpy as np
from saemix.utils import transphi
from typing import List


def saemix_predict(saemix_object, type: List[str] = ["ipred", "ppred"]):
    """
    计算预测值
    
    参数
    -----
    saemix_object : SaemixObject
        SAEM拟合结果对象
    type : List[str]
        预测类型列表: "ipred", "ppred", "ypred", "icpred"
    
    返回
    -----
    dict
        包含不同类型预测值的字典
    """
    
    model = saemix_object.model
    data = saemix_object.data
    results = saemix_object.results
    
    xind = data.data[data.name_predictors].values
    yobs = data.data[data.name_response].values
    index = data.data['index'].values
    
    predictions = {}
    
    if "ppred" in type:
        psiM = transphi(results.mean_phi, model.transform_par)
        if psiM.ndim == 1:
            psiM = psiM.reshape(1, -1)
        
        ppred = np.zeros(len(index))
        for i in range(data.n_subjects):
            mask = (index == i)
            if np.any(mask):
                psi_i = psiM[i, :].reshape(1, -1)
                id_i = np.zeros(np.sum(mask), dtype=int)
                xind_i = xind[mask]
                ppred[mask] = model.model(psi_i, id_i, xind_i)
        
        predictions['ppred'] = ppred
    
    if "ipred" in type:
        if results.map_psi is None:
            from saemix.algorithm.map_estimation import map_saemix
            saemix_object = map_saemix(saemix_object)
            results = saemix_object.results
        
        if isinstance(results.map_psi, np.ndarray):
            map_psi = results.map_psi
        else:
            map_psi = results.map_psi.values
        
        if map_psi.ndim == 1:
            map_psi = map_psi.reshape(1, -1)
        
        ipred = np.zeros(len(index))
        for i in range(data.n_subjects):
            mask = (index == i)
            if np.any(mask):
                psi_i = map_psi[i, :].reshape(1, -1)
                id_i = np.zeros(np.sum(mask), dtype=int)
                xind_i = xind[mask]
                ipred[mask] = model.model(psi_i, id_i, xind_i)
        
        predictions['ipred'] = ipred
    
    if "icpred" in type:
        if results.cond_mean_psi is None:
            results.cond_mean_psi = transphi(results.cond_mean_phi, model.transform_par)
        
        cond_psi = results.cond_mean_psi
        if cond_psi.ndim == 1:
            cond_psi = cond_psi.reshape(1, -1)
        
        icpred = np.zeros(len(index))
        for i in range(data.n_subjects):
            mask = (index == i)
            if np.any(mask):
                psi_i = cond_psi[i, :].reshape(1, -1)
                id_i = np.zeros(np.sum(mask), dtype=int)
                xind_i = xind[mask]
                icpred[mask] = model.model(psi_i, id_i, xind_i)
        
        predictions['icpred'] = icpred
    
    if "ypred" in type:
        if "ppred" not in predictions:
            predictions['ppred'] = saemix_predict(saemix_object, type=["ppred"])['ppred']
        predictions['ypred'] = predictions['ppred']
    
    saemix_object.results = results
    return predictions