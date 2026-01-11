import os
from typing import Dict, List, Optional, Union

import pandas as pd


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
        self.name_genetic_covariates = (
            name_genetic_covariates if name_genetic_covariates else []
        )
        self.name_mdv = name_mdv if name_mdv else ""
        self.name_cens = name_cens if name_cens else ""
        self.name_occ = name_occ if name_occ else ""
        self.name_ytype = name_ytype if name_ytype else ""
        self.units = units if units else {}
        self.verbose = verbose
        self.automatic = automatic
        self.yorig: Optional[pd.Series] = None

        self._load_data()
        self._validate_data()
        self._process_data()

    def _load_data(self):
        if isinstance(self.name_data, pd.DataFrame):
            self.data = self.name_data.copy()
            self.name_data = "DataFrame"
        elif isinstance(self.name_data, str):
            if os.path.exists(self.name_data):
                self.data = pd.read_csv(self.name_data, sep=None, engine="python")
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

        # 修复：使用列表推导式构造新列表，而不是在遍历时修改
        if self.name_covariates:
            valid_covariates = []
            ignored_covariates = []

            for cov in self.name_covariates:
                if cov in self.data.columns:
                    valid_covariates.append(cov)
                else:
                    ignored_covariates.append(cov)

            if ignored_covariates:
                if self.verbose:
                    print(
                        f"Warning: Covariate columns not found, ignoring: {ignored_covariates}"
                    )

            if not valid_covariates and self.verbose:
                print("Warning: No valid covariates found, covariate list is empty")

            if self.verbose and valid_covariates:
                print(f"Valid covariates found: {valid_covariates}")

            self.name_covariates = valid_covariates

    def _process_data(self):
        """处理数据，修复辅助列映射问题"""
        # 构建需要保留的列列表
        all_cols = [self.name_group] + self.name_predictors + [self.name_response]
        if self.name_covariates:
            all_cols.extend(self.name_covariates)

        # 新增：构建辅助列映射并验证列存在性
        auxiliary_mapping = {
            "mdv": self.name_mdv,
            "cens": self.name_cens,
            "occ": self.name_occ,
            "ytype": self.name_ytype,
        }

        auxiliary_cols = []
        for internal_name, user_col in auxiliary_mapping.items():
            if user_col:  # 用户指定了列名
                if user_col in self.data.columns:
                    if user_col not in all_cols:
                        auxiliary_cols.append(user_col)
                    if self.verbose:
                        print(f"Using column '{user_col}' for {internal_name}")
                else:
                    raise ValueError(
                        f"[SaemixData] Specified {internal_name} column '{user_col}' not found in data. "
                        f"Context: available_columns={list(self.data.columns)[:10]}... "
                        f"Suggestion: Check column name spelling or use name_{internal_name}=None for default."
                    )
            elif self.verbose:
                print(f"Using default values for {internal_name}")

        # 保留所有需要的列（包括辅助列）
        all_cols.extend(auxiliary_cols)
        self.data = self.data[all_cols].copy()

        self.data = self.data.sort_values([self.name_group, self.name_predictors[0]])

        unique_ids = self.data[self.name_group].unique()
        self.n_subjects = len(unique_ids)

        id_map = {orig_id: idx for idx, orig_id in enumerate(unique_ids)}
        self.data["index"] = self.data[self.name_group].map(id_map)

        # 处理辅助列：使用原始列名映射到内部列名
        if self.name_mdv:
            self.data["mdv"] = self.data[self.name_mdv]
        else:
            self.data["mdv"] = self.data[self.name_response].isna().astype(int)

        if self.name_cens:
            self.data["cens"] = self.data[self.name_cens]
        else:
            self.data["cens"] = 0

        if self.name_occ:
            self.data["occ"] = self.data[self.name_occ]
        else:
            self.data["occ"] = 1

        if self.name_ytype:
            self.data["ytype"] = self.data[self.name_ytype]
        else:
            self.data["ytype"] = 1

        self.n_ind_obs = self.data.groupby("index").size().values

        valid_rows = self.data["mdv"] == 0
        self.data = self.data[valid_rows].copy()
        self.n_ind_obs = self.data.groupby("index").size().values

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
