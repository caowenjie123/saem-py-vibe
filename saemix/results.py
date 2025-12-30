import numpy as np
from typing import Optional
from saemix.data import SaemixData
from saemix.model import SaemixModel


class SaemixRes:
    def __init__(self):
        self.fixed_effects = None
        self.omega = None
        self.respar = None
        self.ll = None
        self.aic = None
        self.bic = None
        self.ll_is = None
        self.aic_is = None
        self.bic_is = None
        self.ll_gq = None
        self.aic_gq = None
        self.bic_gq = None
        self.npar_est = None
        self.mean_phi = None
        self.map_phi = None
        self.map_psi = None
        self.map_eta = None
        self.cond_mean_phi = None
        self.cond_mean_psi = None
        self.cond_mean_eta = None
        self.cond_var_phi = None


class SaemixObject:
    def __init__(self, data: SaemixData, model: SaemixModel, options: dict):
        self.data = data
        self.model = model
        self.options = options
        self.results = SaemixRes()
    
    def psi(self, type: str = "mode") -> np.ndarray:
        if type == "mode":
            return self.results.map_psi
        else:
            return self.results.cond_mean_psi
    
    def phi(self, type: str = "mode") -> np.ndarray:
        if type == "mode":
            return self.results.map_phi
        else:
            return self.results.cond_mean_phi
    
    def eta(self, type: str = "mode") -> np.ndarray:
        if type == "mode":
            return self.results.map_eta
        else:
            return self.results.cond_mean_eta
    
    def predict(self, type: str = "ipred", newdata: Optional = None) -> np.ndarray:
        """
        预测方法
        
        参数
        -----
        type : str
            预测类型: "ipred" (MAP个体预测), "ppred" (群体预测), 
            "ypred" (群体预测均值), "icpred" (条件均值个体预测)
        newdata : Optional
            新数据（暂不支持）
        
        返回
        -----
        np.ndarray
            预测值
        """
        from saemix.algorithm.predict import saemix_predict
        if newdata is not None:
            raise NotImplementedError("newdata prediction not yet implemented")
        pred_dict = saemix_predict(self, type=[type])
        return pred_dict.get(type, None)
    
    def summary(self):
        print("SaemixObject summary")
        print(f"  Model: {self.model.description}")
        print(f"  Subjects: {self.data.n_subjects}")
        if self.results.fixed_effects is not None:
            print(f"  Fixed effects estimated: {len(self.results.fixed_effects)}")
    
    def __repr__(self):
        return f"SaemixObject:\n  Data: {self.data.n_subjects} subjects\n  Model: {self.model.description}"
