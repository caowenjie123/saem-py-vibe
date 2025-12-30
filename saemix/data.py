import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Any
import os


class SaemixData:
    def __init__(
        self,
        name_data: Union[str, pd.DataFrame],
        name_group: str,
        name_predictors: Union[str, List[str]],
        name_response: str,
        name_X: Optional[str] = None,
        name_covariates: Optional[List[str]] = None,
        name_genetic_covariates: Optional[List[str]] = None,
        name_mdv: Optional[str] = None,
        name_cens: Optional[str] = None,
        name_occ: Optional[str] = None,
        name_ytype: Optional[str] = None,
        units: Optional[Dict[str, Union[str, List[str]]]] = None,
        verbose: bool = True,
        automatic: bool = True,
    ):
        self.name_data = name_data
        self.name_group = name_group
        if isinstance(name_predictors, str):
            self.name_predictors = [name_predictors]
        else:
            self.name_predictors = name_predictors
        self.name_response = name_response
        self.name_X = name_X
        self.name_covariates = name_covariates if name_covariates else []
        self.name_genetic_covariates = name_genetic_covariates if name_genetic_covariates else []
        self.name_mdv = name_mdv if name_mdv else ""
        self.name_cens = name_cens if name_cens else ""
        self.name_occ = name_occ if name_occ else ""
        self.name_ytype = name_ytype if name_ytype else ""
        self.units = units if units else {}
        self.verbose = verbose
        self.automatic = automatic
        self.yorig = None
        
        self._load_data()
        self._validate_data()
        self._process_data()
        
    def _load_data(self):
        if isinstance(self.name_data, pd.DataFrame):
            self.data = self.name_data.copy()
            self.name_data = "DataFrame"
        elif isinstance(self.name_data, str):
            if os.path.exists(self.name_data):
                self.data = pd.read_csv(self.name_data, sep=None, engine='python')
            else:
                raise FileNotFoundError(f"Data file not found: {self.name_data}")
        else:
            raise TypeError("name_data must be a pandas DataFrame or file path string")
    
    def _validate_data(self):
        missing_cols = []
        
        if self.name_group not in self.data.columns:
            missing_cols.append(self.name_group)
        
        for pred in self.name_predictors:
            if pred not in self.data.columns:
                missing_cols.append(pred)
        
        if self.name_response not in self.data.columns:
            missing_cols.append(self.name_response)
        
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")
        
        if self.name_covariates:
            for cov in self.name_covariates:
                if cov not in self.data.columns:
                    if self.verbose:
                        print(f"Warning: Covariate column '{cov}' not found, removing from list")
                    self.name_covariates.remove(cov)
    
    def _process_data(self):
        all_cols = [self.name_group] + self.name_predictors + [self.name_response]
        if self.name_covariates:
            all_cols.extend(self.name_covariates)
        
        self.data = self.data[all_cols].copy()
        
        self.data = self.data.sort_values([self.name_group, self.name_predictors[0]])
        
        unique_ids = self.data[self.name_group].unique()
        self.n_subjects = len(unique_ids)
        
        id_map = {orig_id: idx for idx, orig_id in enumerate(unique_ids)}
        self.data['index'] = self.data[self.name_group].map(id_map)
        
        if self.name_mdv:
            if self.name_mdv in self.data.columns:
                self.data['mdv'] = self.data[self.name_mdv]
            else:
                self.data['mdv'] = 0
        else:
            self.data['mdv'] = self.data[self.name_response].isna().astype(int)
        
        if self.name_cens:
            if self.name_cens in self.data.columns:
                self.data['cens'] = self.data[self.name_cens]
            else:
                self.data['cens'] = 0
        else:
            self.data['cens'] = 0
        
        if self.name_occ:
            if self.name_occ in self.data.columns:
                self.data['occ'] = self.data[self.name_occ]
            else:
                self.data['occ'] = 1
        else:
            self.data['occ'] = 1
        
        if self.name_ytype:
            if self.name_ytype in self.data.columns:
                self.data['ytype'] = self.data[self.name_ytype]
            else:
                self.data['ytype'] = 1
        else:
            self.data['ytype'] = 1
        
        self.n_ind_obs = self.data.groupby('index').size().values
        
        valid_rows = self.data['mdv'] == 0
        self.data = self.data[valid_rows].copy()
        self.n_ind_obs = self.data.groupby('index').size().values
        
        self.n_total_obs = len(self.data)
        
        if not self.name_X:
            self.name_X = self.name_predictors[0]
        
        if self.name_covariates:
            self.ocov = self.data[self.name_covariates].copy()
            for cov in self.name_covariates:
                unique_vals = self.data[cov].dropna().unique()
                if len(unique_vals) == 2:
                    self.data[cov] = pd.Categorical(self.data[cov]).codes
        else:
            self.ocov = pd.DataFrame()
        
        if self.n_subjects < 2 and self.verbose:
            print(f"Warning: Only {self.n_subjects} subject(s) in dataset")
    
    def __repr__(self):
        return f"SaemixData object:\n  Subjects: {self.n_subjects}\n  Total observations: {self.n_total_obs}\n  Predictors: {self.name_predictors}\n  Response: {self.name_response}"


def saemix_data(
    name_data: Union[str, pd.DataFrame],
    name_group: str,
    name_predictors: Union[str, List[str]],
    name_response: str,
    name_X: Optional[str] = None,
    name_covariates: Optional[List[str]] = None,
    name_genetic_covariates: Optional[List[str]] = None,
    name_mdv: Optional[str] = None,
    name_cens: Optional[str] = None,
    name_occ: Optional[str] = None,
    name_ytype: Optional[str] = None,
    units: Optional[Dict[str, Union[str, List[str]]]] = None,
    verbose: bool = True,
    automatic: bool = True,
) -> SaemixData:
    return SaemixData(
        name_data=name_data,
        name_group=name_group,
        name_predictors=name_predictors,
        name_response=name_response,
        name_X=name_X,
        name_covariates=name_covariates,
        name_genetic_covariates=name_genetic_covariates,
        name_mdv=name_mdv,
        name_cens=name_cens,
        name_occ=name_occ,
        name_ytype=name_ytype,
        units=units,
        verbose=verbose,
        automatic=automatic,
    )
