# Python SAEM 接口设计文档

## 1. 项目概述

### 1.1 SAEM 算法简介

Stochastic Approximation Expectation Maximization (SAEM) 算法是用于非线性混合效应模型参数估计的算法。它是标准 EM 算法的随机逼近版本，由 Kuhn 和 Lavielle 提出。

### 1.2 Python 版本的目标和范围

本项目的目标是在 Python 中实现与 R saemix 包功能相同的 SAEM 算法库。主要目标包括：

- 实现完整的 SAEM 算法核心功能
- 提供与 R 版本功能一致的接口
- 适配 Python 的 0-based 索引特性
- 使用 Python 生态系统的标准库（numpy, pandas, scipy 等）

### 1.3 R saemix 包功能概览

R saemix 包提供以下核心功能：

- **数据管理**：SaemixData 类用于管理和验证纵向数据
- **模型定义**：SaemixModel 类用于定义非线性混合效应模型
- **参数估计**：saemix() 函数执行 SAEM 算法进行参数估计
- **结果分析**：提供参数提取（psi, phi, eta）、预测、绘图等功能

## 2. R 与 Python 差异分析

### 2.1 索引差异

**R 使用 1-based 索引，Python 使用 0-based 索引。这是最关键的差异。**

#### 2.1.1 矩阵/数组索引

**R 代码示例：**
```r
model1cpt <- function(psi, id, xidep) { 
  dose <- xidep[,1]      # 第一个预测变量（列索引从1开始）
  tim <- xidep[,2]       # 第二个预测变量
  ka <- psi[id,1]        # 第一个参数（列索引从1开始）
  V <- psi[id,2]         # 第二个参数
  CL <- psi[id,3]        # 第三个参数
  # ...
}
```

**Python 对应代码：**
```python
def model1cpt(psi, id, xidep):
    dose = xidep[:, 0]    # 第一个预测变量（列索引从0开始）
    tim = xidep[:, 1]     # 第二个预测变量
    ka = psi[id, 0]       # 第一个参数（列索引从0开始）
    V = psi[id, 1]        # 第二个参数
    CL = psi[id, 2]       # 第三个参数
    # ...
```

#### 2.1.2 向量索引

R 中向量索引从 1 开始，Python 中从 0 开始。在模型函数中，`id` 参数用于索引个体，需要注意：

- R：`id` 通常是 1, 2, 3, ..., N（N 为个体数）
- Python：`id` 应该是 0, 1, 2, ..., N-1

#### 2.1.3 参数位置索引

在 R 中：
- `transform.par=c(1,1,1)` 表示第1、2、3个参数的变换类型
- `covariate.model` 矩阵的行列索引从1开始

在 Python 中：
- `transform_par=[1,1,1]` 表示第0、1、2个参数的变换类型
- `covariate_model` 矩阵的行列索引从0开始

### 2.2 数据结构差异

| R 类型 | Python 类型 | 说明 |
|--------|-------------|------|
| `data.frame` | `pandas.DataFrame` | 表格数据 |
| `matrix` | `numpy.ndarray` | 矩阵/数组 |
| `list` | `dict` 或 `list` | 列表/字典 |
| `numeric` | `numpy.ndarray` 或 `float` | 数值类型 |
| `character` | `str` 或 `list[str]` | 字符串类型 |
| `logical` | `bool` 或 `numpy.ndarray[bool]` | 布尔类型 |

### 2.3 命名约定

- **R**：使用驼峰命名（camelCase），如 `name.data`, `name.group`, `saemixData`
- **Python**：推荐使用下划线命名（snake_case），如 `name_data`, `name_group`, `saemix_data`

但为保持与 R 版本的兼容性和熟悉度，Python 版本也可以考虑支持驼峰命名作为别名。

### 2.4 函数调用方式

- **R**：使用位置参数和命名参数，支持部分匹配
- **Python**：支持位置参数和关键字参数，推荐使用关键字参数提高可读性

### 2.5 属性访问

- **R**：使用 `object["slot"]` 或 `object@slot` 访问对象属性
- **Python**：使用 `object.attribute` 或 `object.method()` 访问属性或方法

## 3. 核心类设计

### 3.1 SaemixData 类

#### 3.1.1 R 原始接口

```r
saemixData(
  name.data,              # 数据文件名或数据框
  header=TRUE,            # 是否有表头
  sep="",                 # 分隔符
  na="NA",                # 缺失值标识
  name.group,             # 个体ID列名
  name.predictors,        # 预测变量列名
  name.response,          # 响应变量列名
  name.X,                 # 用于绘图的X轴变量名
  name.covariates=c(),    # 协变量列名
  name.genetic.covariates=c(), # 遗传协变量列名
  name.mdv="",            # 缺失数据指示列名
  name.cens="",           # 删失数据指示列名
  name.occ="",            # 事件列名
  name.ytype="",          # 响应类型列名
  units=list(x="", y="", covariates=c()), # 单位
  verbose=TRUE,           # 是否打印消息
  automatic=TRUE          # 是否自动识别列名
)
```

#### 3.1.2 Python 设计

**类定义：**
```python
class SaemixData:
    """
    纵向数据结构类，用于 SAEM 算法
    
    参数
    -----
    name_data : str 或 pandas.DataFrame
        数据文件名或 DataFrame 对象
    name_group : str
        个体ID列名
    name_predictors : str 或 list[str]
        预测变量列名（列表）
    name_response : str
        响应变量列名
    name_X : str, optional
        用于绘图的X轴变量名（默认使用第一个预测变量）
    name_covariates : list[str], optional
        协变量列名列表
    name_genetic_covariates : list[str], optional
        遗传协变量列名列表
    name_mdv : str, optional
        缺失数据指示列名
    name_cens : str, optional
        删失数据指示列名
    name_occ : str, optional
        事件列名
    name_ytype : str, optional
        响应类型列名
    units : dict, optional
        单位字典，格式为 {'x': str, 'y': str, 'covariates': list[str]}
    verbose : bool, default=True
        是否打印消息
    automatic : bool, default=True
        是否自动识别列名
    """
```

**属性映射表：**

| R slot | Python 属性 | 类型 | 说明 |
|--------|-------------|------|------|
| `name.data` | `name_data` | `str` | 数据名称 |
| `header` | (内部使用) | `bool` | 是否有表头 |
| `sep` | (内部使用) | `str` | 分隔符 |
| `na` | (内部使用) | `str` | 缺失值标识 |
| `name.group` | `name_group` | `str` | 个体ID列名 |
| `name.predictors` | `name_predictors` | `list[str]` | 预测变量列名 |
| `name.response` | `name_response` | `str` | 响应变量列名 |
| `name.covariates` | `name_covariates` | `list[str]` | 协变量列名 |
| `name.X` | `name_X` | `str` | X轴变量名 |
| `data` | `data` | `pandas.DataFrame` | 数据表格 |
| `N` | `n_subjects` | `int` | 个体数量 |
| `ntot.obs` | `n_total_obs` | `int` | 总观测数 |
| `nind.obs` | `n_ind_obs` | `numpy.ndarray` | 每个个体的观测数 |
| `units` | `units` | `dict` | 单位信息 |

**主要方法：**
- `__init__()`: 构造函数，验证和初始化数据
- `__repr__()`: 字符串表示
- `validate()`: 数据验证
- `plot()`: 数据可视化（可选）

#### 3.1.3 设计要点

1. **数据容器**：使用 `pandas.DataFrame` 存储数据，便于数据处理和分析
2. **列名指定**：通过字符串指定列名（而非 R 中的列位置索引），更符合 Python 习惯
3. **索引处理**：内部统一使用 0-based 索引，个体ID在内部重新映射为 0, 1, 2, ...

### 3.2 SaemixModel 类

#### 3.2.1 R 原始接口

```r
saemixModel(
  model,                  # 模型函数
  psi0,                   # 初始参数矩阵
  description="",         # 模型描述
  modeltype="structural", # 模型类型：structural 或 likelihood
  name.response="",       # 响应变量名
  name.sigma=character(), # 误差参数名
  error.model="constant", # 误差模型类型
  transform.par=numeric(),# 参数变换类型（0=normal, 1=log-normal, 2=probit, 3=logit）
  fixed.estim=numeric(),  # 参数是否估计（1=估计，0=固定）
  covariate.model=matrix(),# 协变量模型矩阵
  covariance.model=matrix(),# 协方差模型矩阵
  omega.init=matrix(),    # Omega 初始值矩阵
  error.init=numeric(),   # 误差模型初始值
  name.modpar=character(),# 模型参数名
  verbose=TRUE
)
```

#### 3.2.2 Python 设计

**类定义：**
```python
class SaemixModel:
    """
    非线性混合效应模型定义类
    
    参数
    -----
    model : callable
        模型函数，必须接受三个参数：psi, id, xidep
        - psi: (n_individuals, n_parameters) numpy.ndarray，个体参数矩阵
        - id: (n_obs,) numpy.ndarray，个体索引（0-based）
        - xidep: (n_obs, n_predictors) numpy.ndarray，预测变量矩阵
        返回: (n_obs,) numpy.ndarray，预测值或对数似然
    psi0 : numpy.ndarray 或 list
        初始参数矩阵，形状为 (n_rows, n_parameters)
        - 如果只有一行，表示总体均值参数
        - 如果有多行，第一行是总体均值，后续行是协变量效应
    description : str, default=""
        模型描述
    modeltype : str, default="structural"
        模型类型："structural" 或 "likelihood"
    error_model : str 或 list[str], default="constant"
        误差模型类型："constant", "proportional", "combined", "exponential"
    transform_par : list[int], optional
        参数变换类型列表（0=normal, 1=log-normal, 2=probit, 3=logit）
    fixed_estim : list[int], optional
        参数是否估计（1=估计，0=固定）
    covariate_model : numpy.ndarray, optional
        协变量模型矩阵，形状为 (n_covariates, n_parameters)
    covariance_model : numpy.ndarray, optional
        协方差模型矩阵，形状为 (n_parameters, n_parameters)
    omega_init : numpy.ndarray, optional
        Omega 初始值矩阵，形状为 (n_parameters, n_parameters)
    error_init : list[float], optional
        误差模型初始值
    name_modpar : list[str], optional
        模型参数名称列表
    verbose : bool, default=True
        是否打印消息
    """
```

**模型函数示例（Python，0-based 索引）：**

```python
def model1cpt(psi, id, xidep):
    """
    一室模型，一级吸收
    
    参数
    -----
    psi : (n_individuals, 3) ndarray
        个体参数 [ka, V, CL]，索引从0开始
    id : (n_obs,) ndarray
        个体索引，从0开始
    xidep : (n_obs, 2) ndarray
        预测变量 [dose, time]，索引从0开始
    
    返回
    -----
    ypred : (n_obs,) ndarray
        预测值
    """
    dose = xidep[:, 0]  # 第0列：剂量
    tim = xidep[:, 1]   # 第1列：时间
    ka = psi[id, 0]     # 第0个参数：吸收速率常数
    V = psi[id, 1]      # 第1个参数：分布容积
    CL = psi[id, 2]     # 第2个参数：清除率
    k = CL / V
    ypred = dose * ka / (V * (ka - k)) * (np.exp(-k * tim) - np.exp(-ka * tim))
    return ypred
```

**属性映射表：**

| R slot | Python 属性 | 类型 | 说明 |
|--------|-------------|------|------|
| `model` | `model` | `callable` | 模型函数 |
| `description` | `description` | `str` | 模型描述 |
| `modeltype` | `modeltype` | `str` | 模型类型 |
| `psi0` | `psi0` | `numpy.ndarray` | 初始参数矩阵 |
| `transform.par` | `transform_par` | `numpy.ndarray` | 参数变换类型 |
| `fixed.estim` | `fixed_estim` | `numpy.ndarray` | 是否估计 |
| `error.model` | `error_model` | `str` 或 `list[str]` | 误差模型 |
| `covariate.model` | `covariate_model` | `numpy.ndarray` | 协变量模型矩阵 |
| `covariance.model` | `covariance_model` | `numpy.ndarray` | 协方差模型矩阵 |
| `omega.init` | `omega_init` | `numpy.ndarray` | Omega 初始值 |
| `error.init` | `error_init` | `numpy.ndarray` | 误差初始值 |
| `nb.parameters` | `n_parameters` | `int` | 参数数量 |
| `name.modpar` | `name_modpar` | `list[str]` | 参数名称 |

#### 3.2.3 设计要点

1. **模型函数签名**：
   - `psi`: (n_individuals, n_parameters) 数组，索引从 0 开始
   - `id`: 个体索引数组，从 0 开始（对应 R 中的 1, 2, ..., N）
   - `xidep`: (n_obs, n_predictors) 数组，索引从 0 开始

2. **psi0 矩阵格式**：
   - Python 使用 0-based 索引
   - 第一行（索引0）是总体均值参数
   - 后续行是协变量效应

3. **矩阵参数索引**：
   - `covariate_model`, `covariance_model`, `omega_init` 都使用 0-based 索引

### 3.3 SaemixObject 类（结果对象）

#### 3.3.1 R 原始接口

`SaemixObject` 由 `saemix()` 函数返回，包含以下 slots：
- `data`: SaemixData 对象
- `model`: SaemixModel 对象
- `results`: SaemixRes 对象（拟合结果）
- `rep.data`: SaemixRepData 对象（算法内部数据）
- `sim.data`: SaemixSimData 对象（模拟数据）
- `options`: 算法选项列表
- `prefs`: 绘图选项列表

#### 3.3.2 Python 设计

```python
class SaemixObject:
    """
    SAEM 算法拟合结果对象
    
    属性
    -----
    data : SaemixData
        输入数据对象
    model : SaemixModel
        模型对象
    results : SaemixRes
        拟合结果对象
    options : dict
        算法选项
    """
    
    def psi(self, type="mode"):
        """
        提取个体参数 psi
        
        参数
        -----
        type : str, default="mode"
            "mode" 或 "mean"，使用条件分布的模式或均值
        
        返回
        -----
        numpy.ndarray
            个体参数矩阵，形状为 (n_subjects, n_parameters)
        """
        pass
    
    def phi(self, type="mode"):
        """提取个体参数 phi（未变换的参数）"""
        pass
    
    def eta(self, type="mode"):
        """提取随机效应 eta"""
        pass
    
    def predict(self, type="ipred"):
        """
        预测
        
        参数
        -----
        type : str
            "ipred", "ypred", "ppred", "icpred"
        
        返回
        -----
        numpy.ndarray
            预测值
        """
        pass
    
    def summary(self):
        """打印结果摘要"""
        pass
    
    def __repr__(self):
        """字符串表示"""
        pass
```

**属性访问方式对比：**

| R | Python |
|---|--------|
| `object["data"]` | `object.data` |
| `object["model"]` | `object.model` |
| `object["results"]` | `object.results` |
| `psi(object)` | `object.psi()` |
| `phi(object)` | `object.phi()` |
| `eta(object)` | `object.eta()` |
| `predict(object)` | `object.predict()` |

### 3.4 SaemixControl（控制参数）

#### 3.4.1 R 原始接口

```r
saemixControl(
  map=TRUE,                    # 是否计算 MAP 估计
  fim=TRUE,                    # 是否计算 Fisher 信息矩阵
  ll.is=TRUE,                  # 是否计算重要性采样似然
  ll.gq=FALSE,                 # 是否计算高斯积分似然
  nbiter.saemix=c(300,100),    # SAEM 迭代次数 [K1, K2]
  nbiter.sa=NA,                # 模拟退火迭代次数
  nb.chains=1,                 # 链数
  fix.seed=TRUE,               # 是否固定随机种子
  seed=23456,                  # 随机种子
  nmc.is=5000,                 # 重要性采样样本数
  nu.is=4,                     # 重要性采样自由度
  print.is=FALSE,              # 是否打印 IS 信息
  nbdisplay=100,               # 显示频率
  displayProgress=FALSE,       # 是否显示进度
  nbiter.burn=5,               # 燃烧期迭代次数
  nbiter.map=5,                # MAP 迭代次数
  nbiter.mcmc=c(2,2,2,0),      # MCMC 迭代次数
  proba.mcmc=0.4,              # MCMC 接受概率
  stepsize.rw=0.4,             # 随机游走步长
  rw.init=0.5,                 # 随机游走初始值
  alpha.sa=0.97,               # 模拟退火衰减系数
  nnodes.gq=12,                # 高斯积分节点数
  nsd.gq=4,                    # 高斯积分标准差倍数
  maxim.maxiter=100,           # 最大化迭代次数
  nb.sim=1000,                 # 模拟次数
  nb.simpred=100,              # 预测模拟次数
  ipar.lmcmc=50,               # 个体参数 MCMC 长度
  ipar.rmcmc=0.05,             # 个体参数 MCMC 比率
  print=TRUE,                  # 是否打印
  save=TRUE,                   # 是否保存结果
  save.graphs=TRUE,            # 是否保存图形
  directory="newdir",          # 输出目录
  warnings=FALSE               # 是否显示警告
)
```

#### 3.4.2 Python 设计

**方案1：使用字典（推荐用于函数参数）**
```python
def saemix_control(
    map=True,
    fim=True,
    ll_is=True,
    ll_gq=False,
    nbiter_saemix=(300, 100),
    nbiter_sa=None,
    nb_chains=1,
    fix_seed=True,
    seed=23456,
    nmc_is=5000,
    nu_is=4,
    print_is=False,
    nbdisplay=100,
    display_progress=False,
    nbiter_burn=5,
    nbiter_map=5,
    nbiter_mcmc=(2, 2, 2, 0),
    proba_mcmc=0.4,
    stepsize_rw=0.4,
    rw_init=0.5,
    alpha_sa=0.97,
    nnodes_gq=12,
    nsd_gq=4,
    maxim_maxiter=100,
    nb_sim=1000,
    nb_simpred=100,
    ipar_lmcmc=50,
    ipar_rmcmc=0.05,
    print_results=True,
    save=True,
    save_graphs=True,
    directory="newdir",
    warnings=False,
    **kwargs
) -> dict:
    """
    创建 SAEM 算法控制参数字典
    
    返回
    -----
    dict
        控制参数字典
    """
    control = {
        'map': map,
        'fim': fim,
        'll_is': ll_is,
        'll_gq': ll_gq,
        'nbiter_saemix': nbiter_saemix,
        'nbiter_sa': nbiter_sa if nbiter_sa is not None else nbiter_saemix[0] // 2,
        'nb_chains': nb_chains,
        'fix_seed': fix_seed,
        'seed': seed,
        # ... 其他参数
    }
    control.update(kwargs)
    return control
```

**方案2：使用 dataclass（可选，用于类型提示）**
```python
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class SaemixControl:
    map: bool = True
    fim: bool = True
    ll_is: bool = True
    ll_gq: bool = False
    nbiter_saemix: Tuple[int, int] = (300, 100)
    nbiter_sa: Optional[int] = None
    nb_chains: int = 1
    fix_seed: bool = True
    seed: int = 23456
    # ... 其他字段
    
    def to_dict(self):
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items()}
```

**参数命名映射表：**

| R 参数 | Python 参数 | 类型 | 说明 |
|--------|-------------|------|------|
| `map` | `map` | `bool` | 是否计算 MAP |
| `fim` | `fim` | `bool` | 是否计算 FIM |
| `ll.is` | `ll_is` | `bool` | 是否计算 IS 似然 |
| `ll.gq` | `ll_gq` | `bool` | 是否计算 GQ 似然 |
| `nbiter.saemix` | `nbiter_saemix` | `tuple[int, int]` | SAEM 迭代次数 |
| `nb.chains` | `nb_chains` | `int` | 链数 |
| `fix.seed` | `fix_seed` | `bool` | 是否固定种子 |
| `seed` | `seed` | `int` | 随机种子 |
| `displayProgress` | `display_progress` | `bool` | 是否显示进度 |
| `nb.sim` | `nb_sim` | `int` | 模拟次数 |
| `nb.simpred` | `nb_simpred` | `int` | 预测模拟次数 |

## 4. 主要接口函数设计

### 4.1 主函数 `saemix()`

```python
def saemix(
    model: SaemixModel,
    data: SaemixData,
    control: Optional[dict] = None
) -> SaemixObject:
    """
    执行 SAEM 算法进行参数估计
    
    参数
    -----
    model : SaemixModel
        模型对象
    data : SaemixData
        数据对象
    control : dict, optional
        控制参数字典，如果为 None 则使用默认参数
    
    返回
    -----
    SaemixObject
        拟合结果对象
    
    示例
    -----
    >>> from saemix import saemix, saemix_data, saemix_model, saemix_control
    >>> data = saemix_data(...)
    >>> model = saemix_model(...)
    >>> control = saemix_control(seed=632545, display_progress=True)
    >>> result = saemix(model, data, control)
    """
    if control is None:
        control = saemix_control()
    
    # 参数验证
    if not isinstance(model, SaemixModel):
        raise TypeError("model must be a SaemixModel instance")
    if not isinstance(data, SaemixData):
        raise TypeError("data must be a SaemixData instance")
    
    # 执行算法...
    # 返回 SaemixObject
    pass
```

**与 R 的对比：**

| R | Python |
|---|--------|
| `saemix(model, data, control=list())` | `saemix(model, data, control=None)` |
| `control` 是命名列表 | `control` 是字典 |

### 4.2 数据创建函数 `saemix_data()`

```python
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
    automatic: bool = True
) -> SaemixData:
    """
    创建 SaemixData 对象
    
    参数
    -----
    name_data : str 或 pandas.DataFrame
        数据文件名或 DataFrame 对象
    name_group : str
        个体ID列名
    name_predictors : str 或 list[str]
        预测变量列名（列表）
    name_response : str
        响应变量列名
    name_X : str, optional
        用于绘图的X轴变量名（默认使用第一个预测变量）
    name_covariates : list[str], optional
        协变量列名列表
    name_genetic_covariates : list[str], optional
        遗传协变量列名列表
    name_mdv : str, optional
        缺失数据指示列名
    name_cens : str, optional
        删失数据指示列名
    name_occ : str, optional
        事件列名
    name_ytype : str, optional
        响应类型列名
    units : dict, optional
        单位字典，格式为 {'x': str, 'y': str, 'covariates': list[str]}
    verbose : bool, default=True
        是否打印消息
    automatic : bool, default=True
        是否自动识别列名
    
    返回
    -----
    SaemixData
        数据对象
    
    示例
    -----
    >>> import pandas as pd
    >>> df = pd.read_csv("data.csv")
    >>> data = saemix_data(
    ...     name_data=df,
    ...     name_group="Id",
    ...     name_predictors=["Dose", "Time"],
    ...     name_response="Concentration",
    ...     name_covariates=["Weight", "Sex"],
    ...     units={'x': 'hr', 'y': 'mg/L', 'covariates': ['kg', '-']},
    ...     name_X="Time"
    ... )
    """
    pass
```

**与 R 的对比：**

| R | Python |
|---|--------|
| `saemixData(name.data=..., name.group=c("Id"), ...)` | `saemix_data(name_data=..., name_group="Id", ...)` |
| 列名可以用位置索引 | 列名只能用字符串指定 |
| `units=list(x="hr", y="mg/L", covariates=c("kg","-"))` | `units={'x': 'hr', 'y': 'mg/L', 'covariates': ['kg', '-']}` |

### 4.3 模型创建函数 `saemix_model()`

```python
def saemix_model(
    model: Callable,
    psi0: Union[np.ndarray, List[List[float]], Dict[str, float]],
    description: str = "",
    modeltype: str = "structural",
    error_model: Union[str, List[str]] = "constant",
    transform_par: Optional[List[int]] = None,
    fixed_estim: Optional[List[int]] = None,
    covariate_model: Optional[np.ndarray] = None,
    covariance_model: Optional[np.ndarray] = None,
    omega_init: Optional[np.ndarray] = None,
    error_init: Optional[List[float]] = None,
    name_modpar: Optional[List[str]] = None,
    verbose: bool = True
) -> SaemixModel:
    """
    创建 SaemixModel 对象
    
    参数
    -----
    model : callable
        模型函数，签名：model(psi, id, xidep) -> ypred
        - psi: (n_individuals, n_parameters) ndarray
        - id: (n_obs,) ndarray，个体索引（0-based）
        - xidep: (n_obs, n_predictors) ndarray
    psi0 : numpy.ndarray 或 list 或 dict
        初始参数矩阵或向量或字典
        - 如果为数组：形状为 (n_rows, n_parameters)
        - 如果为列表：将被转换为数组
        - 如果为字典：键为参数名，值为初始值
    description : str, default=""
        模型描述
    modeltype : str, default="structural"
        模型类型："structural" 或 "likelihood"
    error_model : str 或 list[str], default="constant"
        误差模型类型
    transform_par : list[int], optional
        参数变换类型列表
    fixed_estim : list[int], optional
        参数是否估计（1=估计，0=固定）
    covariate_model : numpy.ndarray, optional
        协变量模型矩阵
    covariance_model : numpy.ndarray, optional
        协方差模型矩阵
    omega_init : numpy.ndarray, optional
        Omega 初始值矩阵
    error_init : list[float], optional
        误差模型初始值
    name_modpar : list[str], optional
        模型参数名称列表
    verbose : bool, default=True
        是否打印消息
    
    返回
    -----
    SaemixModel
        模型对象
    
    示例
    -----
    >>> def model1cpt(psi, id, xidep):
    ...     dose = xidep[:, 0]
    ...     tim = xidep[:, 1]
    ...     ka = psi[id, 0]
    ...     V = psi[id, 1]
    ...     CL = psi[id, 2]
    ...     k = CL / V
    ...     ypred = dose * ka / (V * (ka - k)) * (np.exp(-k * tim) - np.exp(-ka * tim))
    ...     return ypred
    >>> 
    >>> model = saemix_model(
    ...     model=model1cpt,
    ...     psi0=[[1.0, 20.0, 0.5], [0.1, 0.0, -0.01]],
    ...     description="One-compartment model",
    ...     transform_par=[1, 1, 1],
    ...     fixed_estim=[1, 1, 1],
    ...     covariance_model=np.eye(3),
    ...     omega_init=np.eye(3),
    ...     error_model="constant"
    ... )
    """
    pass
```

**与 R 的对比：**

| R | Python |
|---|--------|
| `saemixModel(model=model1cpt, psi0=matrix(...), ...)` | `saemix_model(model=model1cpt, psi0=[[...]], ...)` |
| `psi0` 是 matrix，列索引从1开始 | `psi0` 是 ndarray，索引从0开始 |
| 矩阵使用 `matrix(c(...), ncol=3, byrow=TRUE)` | 数组使用 `np.array([[...]])` 或列表 |

### 4.4 控制参数函数 `saemix_control()`

已在 3.4.2 节详细说明，这里不再重复。

## 5. 索引转换策略

### 5.1 核心原则

**Python 版本统一使用 0-based 索引**，所有用户接口和内部实现都遵循这一原则。

### 5.2 用户模型函数

用户在定义模型函数时，必须使用 0-based 索引：

```python
# 正确：Python 0-based 索引
def model1cpt(psi, id, xidep):
    dose = xidep[:, 0]    # 第0列
    tim = xidep[:, 1]     # 第1列
    ka = psi[id, 0]       # 第0个参数
    V = psi[id, 1]        # 第1个参数
    CL = psi[id, 2]       # 第2个参数
    # ...
```

```r
# R 版本（1-based 索引）
model1cpt <- function(psi, id, xidep) {
    dose <- xidep[, 1]    # 第1列
    tim <- xidep[, 2]     # 第2列
    ka <- psi[id, 1]      # 第1个参数
    V <- psi[id, 2]       # 第2个参数
    CL <- psi[id, 3]      # 第3个参数
    # ...
}
```

### 5.3 内部数据处理

所有内部数据处理统一使用 0-based 索引：

1. **个体ID映射**：
   - 输入数据中的个体ID可以是任意值（如 1, 2, 3, ... 或 'A', 'B', 'C', ...）
   - 内部映射为连续的整数索引：0, 1, 2, ..., N-1
   - 模型函数接收的 `id` 参数是 0-based 索引

2. **参数索引**：
   - `psi0` 矩阵的列索引从 0 开始
   - `transform_par`, `fixed_estim` 等列表的索引从 0 开始
   - `covariate_model`, `covariance_model` 等矩阵的行列索引从 0 开始

3. **预测变量索引**：
   - `xidep` 矩阵的列索引从 0 开始
   - 第一个预测变量索引为 0，第二个为 1，以此类推

### 5.4 与 R 的兼容性

如果需要从 R 迁移代码到 Python：

1. **模型函数转换**：
   - 将所有矩阵/数组索引减 1
   - `xidep[,1]` → `xidep[:, 0]`
   - `psi[id,1]` → `psi[id, 0]`

2. **参数矩阵转换**：
   - `psi0` 矩阵的列索引不变（因为使用列名），但内部访问时使用 0-based
   - `covariate_model` 等矩阵需要检查是否使用了硬编码的行列索引

3. **提供转换工具（可选）**：
   - 可以提供辅助函数帮助从 R 代码生成 Python 代码模板
   - 但这不在当前设计文档的范围内

### 5.5 索引转换检查清单

在实现时需要检查以下位置：

- [ ] 模型函数中的数组索引（`xidep`, `psi`）
- [ ] 个体ID的映射（从原始ID到内部索引）
- [ ] 矩阵访问（`covariate_model`, `covariance_model`, `omega_init`）
- [ ] 列表索引（`transform_par`, `fixed_estim`）
- [ ] 循环索引（`for i in range(n)` 从 0 开始）
- [ ] 切片操作（`array[0:n]` 不包括 n）

## 6. 辅助方法和工具函数

### 6.1 参数提取方法

#### 6.1.1 `psi()` 方法

```python
def psi(self, type: str = "mode") -> np.ndarray:
    """
    提取个体参数 psi（变换后的参数）
    
    参数
    -----
    type : str, default="mode"
        "mode" 或 "mean"，使用条件分布的模式或均值
    
    返回
    -----
    numpy.ndarray
        个体参数矩阵，形状为 (n_subjects, n_parameters)
    """
```

**与 R 的对比：**
- R: `psi(saemix.fit, type="mode")`
- Python: `saemix_result.psi(type="mode")`

#### 6.1.2 `phi()` 方法

```python
def phi(self, type: str = "mode") -> np.ndarray:
    """
    提取个体参数 phi（未变换的参数）
    """
```

#### 6.1.3 `eta()` 方法

```python
def eta(self, type: str = "mode") -> np.ndarray:
    """
    提取随机效应 eta
    """
```

### 6.2 预测方法

```python
def predict(
    self,
    type: str = "ipred",
    newdata: Optional[pd.DataFrame] = None
) -> np.ndarray:
    """
    预测
    
    参数
    -----
    type : str, default="ipred"
        预测类型：
        - "ipred": 个体预测（使用个体参数估计）
        - "ypred": 总体预测（使用总体参数）
        - "ppred": 总体预测（考虑随机效应分布）
        - "icpred": 条件个体预测
    newdata : pandas.DataFrame, optional
        新数据，如果为 None 则使用原始数据
    
    返回
    -----
    numpy.ndarray
        预测值数组
    """
```

**与 R 的对比：**
- R: `predict(saemix.fit, type="ipred")`
- Python: `saemix_result.predict(type="ipred")`

### 6.3 结果摘要方法

```python
def summary(self) -> None:
    """
    打印结果摘要
    
    输出包括：
    - 模型信息
    - 参数估计值
    - 标准误
    - 对数似然值
    - 信息准则（AIC, BIC）
    """
```

**与 R 的对比：**
- R: `summary(saemix.fit)` 或 `print(saemix.fit)`
- Python: `saemix_result.summary()` 或 `print(saemix_result)`

### 6.4 绘图方法（可选）

```python
def plot(
    self,
    plot_type: str = "default",
    **kwargs
) -> None:
    """
    绘制诊断图
    
    参数
    -----
    plot_type : str
        绘图类型：
        - "data": 数据图
        - "convergence": 收敛图
        - "fits": 拟合图
        - "residuals": 残差图
        - "vpc": 可视化预测检查
        - 等等
    **kwargs
        其他绘图参数
    """
```

**说明：** 绘图功能可以使用 `matplotlib` 或 `plotly` 实现，也可以只提供数据提取方法，让用户自行绘图。

## 7. 使用示例对比

### 7.1 数据加载示例

**R 版本：**
```r
library(saemix)
data(theo.saemix)

saemix.data <- saemixData(
  name.data=theo.saemix,
  header=TRUE,
  sep=" ",
  na=NA,
  name.group=c("Id"),
  name.predictors=c("Dose","Time"),
  name.response=c("Concentration"),
  name.covariates=c("Weight","Sex"),
  units=list(x="hr", y="mg/L", covariates=c("kg","-")),
  name.X="Time"
)
```

**Python 版本：**
```python
import pandas as pd
from saemix import saemix_data

# 假设数据已加载到 DataFrame
# theo_data = pd.read_csv("theo.saemix.csv")  # 或从其他来源

data = saemix_data(
    name_data=theo_data,  # 或使用文件名: "theo.saemix.csv"
    name_group="Id",
    name_predictors=["Dose", "Time"],
    name_response="Concentration",
    name_covariates=["Weight", "Sex"],
    units={'x': 'hr', 'y': 'mg/L', 'covariates': ['kg', '-']},
    name_X="Time"
)
```

### 7.2 模型定义示例

**R 版本：**
```r
model1cpt <- function(psi, id, xidep) { 
  dose <- xidep[,1]      # 第1列：剂量
  tim <- xidep[,2]       # 第2列：时间
  ka <- psi[id,1]        # 第1个参数：ka
  V <- psi[id,2]         # 第2个参数：V
  CL <- psi[id,3]        # 第3个参数：CL
  k <- CL/V
  ypred <- dose*ka/(V*(ka-k))*(exp(-k*tim)-exp(-ka*tim))
  return(ypred)
}

saemix.model <- saemixModel(
  model=model1cpt,
  modeltype="structural",
  description="One-compartment model with first-order absorption",
  psi0=matrix(c(1.,20,0.5,0.1,0,-0.01), ncol=3, byrow=TRUE,
              dimnames=list(NULL, c("ka","V","CL"))),
  transform.par=c(1,1,1),
  covariate.model=matrix(c(0,1,0,0,0,0), ncol=3, byrow=TRUE),
  fixed.estim=c(1,1,1),
  covariance.model=matrix(c(1,0,0,0,1,0,0,0,1), ncol=3, byrow=TRUE),
  omega.init=matrix(c(1,0,0,0,1,0,0,0,1), ncol=3, byrow=TRUE),
  error.model="constant"
)
```

**Python 版本：**
```python
import numpy as np
from saemix import saemix_model

def model1cpt(psi, id, xidep):
    """
    一室模型，一级吸收
    
    注意：所有索引从 0 开始
    """
    dose = xidep[:, 0]    # 第0列：剂量
    tim = xidep[:, 1]     # 第1列：时间
    ka = psi[id, 0]       # 第0个参数：ka
    V = psi[id, 1]        # 第1个参数：V
    CL = psi[id, 2]       # 第2个参数：CL
    k = CL / V
    ypred = dose * ka / (V * (ka - k)) * (np.exp(-k * tim) - np.exp(-ka * tim))
    return ypred

model = saemix_model(
    model=model1cpt,
    modeltype="structural",
    description="One-compartment model with first-order absorption",
    psi0=np.array([
        [1.0, 20.0, 0.5],      # 第一行：总体均值参数
        [0.1, 0.0, -0.01]      # 第二行：协变量效应（如果有）
    ]),
    name_modpar=["ka", "V", "CL"],  # 参数名称
    transform_par=[1, 1, 1],        # 所有参数使用 log-normal 变换
    fixed_estim=[1, 1, 1],          # 所有参数都估计
    covariate_model=np.array([      # 协变量模型矩阵（2个协变量 × 3个参数）
        [0, 1, 0],                  # 第0个协变量（Weight）影响第1个参数（V）
        [0, 0, 0]                   # 第1个协变量（Sex）不影响任何参数
    ]),
    covariance_model=np.eye(3),     # 单位矩阵（无协方差）
    omega_init=np.eye(3),           # Omega 初始值
    error_model="constant"
)
```

**关键差异：**
1. 索引从 0 开始：`xidep[:, 0]`, `psi[id, 0]` 等
2. 使用 `numpy.ndarray` 而不是 R 的 `matrix`
3. 使用列表和字典而不是 R 的向量和列表

### 7.3 完整工作流示例

**R 版本：**
```r
library(saemix)
data(theo.saemix)

# 1. 创建数据对象
saemix.data <- saemixData(
  name.data=theo.saemix,
  header=TRUE,
  name.group=c("Id"),
  name.predictors=c("Dose","Time"),
  name.response=c("Concentration"),
  name.covariates=c("Weight","Sex"),
  units=list(x="hr", y="mg/L", covariates=c("kg","-")),
  name.X="Time"
)

# 2. 定义模型函数
model1cpt <- function(psi, id, xidep) { 
  dose <- xidep[,1]
  tim <- xidep[,2]
  ka <- psi[id,1]
  V <- psi[id,2]
  CL <- psi[id,3]
  k <- CL/V
  ypred <- dose*ka/(V*(ka-k))*(exp(-k*tim)-exp(-ka*tim))
  return(ypred)
}

# 3. 创建模型对象
saemix.model <- saemixModel(
  model=model1cpt,
  description="One-compartment model",
  psi0=matrix(c(1.,20,0.5), ncol=3, byrow=TRUE,
              dimnames=list(NULL, c("ka","V","CL"))),
  transform.par=c(1,1,1),
  covariance.model=matrix(c(1,0,0,0,1,0,0,0,1), ncol=3, byrow=TRUE),
  omega.init=matrix(c(1,0,0,0,1,0,0,0,1), ncol=3, byrow=TRUE),
  error.model="constant"
)

# 4. 设置控制参数
saemix.options <- list(
  seed=632545,
  save=FALSE,
  save.graphs=FALSE,
  print=FALSE
)

# 5. 执行拟合
saemix.fit <- saemix(saemix.model, saemix.data, saemix.options)

# 6. 查看结果
print(saemix.fit)
summary(saemix.fit)
psi(saemix.fit)
plot(saemix.fit)
```

**Python 版本：**
```python
import numpy as np
import pandas as pd
from saemix import (
    saemix, saemix_data, saemix_model, saemix_control
)

# 1. 创建数据对象
# 假设数据已加载到 DataFrame
data = saemix_data(
    name_data=theo_data,  # 或使用文件名
    name_group="Id",
    name_predictors=["Dose", "Time"],
    name_response="Concentration",
    name_covariates=["Weight", "Sex"],
    units={'x': 'hr', 'y': 'mg/L', 'covariates': ['kg', '-']},
    name_X="Time"
)

# 2. 定义模型函数
def model1cpt(psi, id, xidep):
    dose = xidep[:, 0]    # 注意：索引从 0 开始
    tim = xidep[:, 1]
    ka = psi[id, 0]
    V = psi[id, 1]
    CL = psi[id, 2]
    k = CL / V
    ypred = dose * ka / (V * (ka - k)) * (np.exp(-k * tim) - np.exp(-ka * tim))
    return ypred

# 3. 创建模型对象
model = saemix_model(
    model=model1cpt,
    description="One-compartment model",
    psi0=np.array([[1.0, 20.0, 0.5]]),  # 单行矩阵
    name_modpar=["ka", "V", "CL"],
    transform_par=[1, 1, 1],
    covariance_model=np.eye(3),
    omega_init=np.eye(3),
    error_model="constant"
)

# 4. 设置控制参数
control = saemix_control(
    seed=632545,
    save=False,
    save_graphs=False,
    print_results=False
)

# 5. 执行拟合
result = saemix(model, data, control)

# 6. 查看结果
print(result)
result.summary()
individual_params = result.psi()
result.plot()  # 如果实现了绘图功能
```

### 7.4 参数提取示例

**R 版本：**
```r
# 提取个体参数（使用 MAP 估计）
psi_est <- psi(saemix.fit, type="mode")

# 提取随机效应（使用均值）
eta_est <- eta(saemix.fit, type="mean")
```

**Python 版本：**
```python
# 提取个体参数（使用 MAP 估计）
psi_est = result.psi(type="mode")

# 提取随机效应（使用均值）
eta_est = result.eta(type="mean")
```

## 8. 实现注意事项

### 8.1 依赖库

Python 版本的主要依赖：

- **numpy**: 数值计算和数组操作
- **pandas**: 数据处理和表格操作
- **scipy**: 科学计算（优化、统计等）
- **matplotlib** (可选): 绘图功能
- **typing**: 类型提示

建议的最小版本：
```
numpy >= 1.20.0
pandas >= 1.3.0
scipy >= 1.7.0
```

### 8.2 性能考虑

1. **数组操作**：优先使用 numpy 的向量化操作，避免 Python 循环
2. **内存管理**：对于大型数据集，注意内存使用，考虑分块处理
3. **算法实现**：核心算法应使用高效的数值计算库

### 8.3 错误处理策略

1. **输入验证**：
   - 检查数据格式和类型
   - 验证参数维度匹配
   - 检查必要字段是否存在

2. **算法执行**：
   - 捕获数值错误（如奇异矩阵、收敛失败等）
   - 提供有意义的错误消息
   - 记录警告信息

3. **用户友好性**：
   - 提供清晰的错误提示
   - 建议解决方案（如果可能）

### 8.4 测试策略

1. **单元测试**：
   - 测试每个类和方法的正确性
   - 测试索引转换的正确性
   - 测试边界情况

2. **集成测试**：
   - 测试完整工作流
   - 与 R 版本结果对比（在相同数据上）

3. **回归测试**：
   - 确保新更改不破坏现有功能
   - 维护测试用例库

### 8.5 文档和示例

1. **API 文档**：使用 docstring 提供详细的函数/类文档
2. **用户指南**：提供使用教程和最佳实践
3. **示例代码**：提供常见用例的示例
4. **迁移指南**：帮助 R 用户迁移到 Python 版本

## 9. 总结

本文档描述了 Python SAEM 算法库的接口设计，重点关注：

1. **索引转换**：从 R 的 1-based 索引转换为 Python 的 0-based 索引
2. **接口一致性**：保持与 R 版本功能一致，但遵循 Python 的编程习惯
3. **数据类型**：使用 Python 生态系统的标准库（numpy, pandas）
4. **用户友好性**：提供清晰的接口和文档

关键设计决策：
- 统一使用 0-based 索引（包括用户模型函数）
- 使用 pandas DataFrame 存储数据
- 使用 numpy ndarray 进行数值计算
- 方法调用风格（`object.method()` 而非 `function(object)`）
- 参数命名使用 snake_case（但可以支持 camelCase 别名）
