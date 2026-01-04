# Design Document: Python saemix Enhancement

## Overview

本设计文档描述了 Python 版本 saemix 库的功能增强实现方案。主要目标是将 Python 版本的功能与 R 版本对齐，实现条件分布估计、模型比较、逐步回归等核心功能。

设计遵循以下原则：
- 与现有代码架构保持一致
- 使用 NumPy/SciPy 进行数值计算
- 使用 Pandas 进行数据管理
- 使用 Matplotlib 进行可视化
- 提供清晰的 API 接口

## Architecture

```
saemix/
├── algorithm/
│   ├── saem.py           # 核心 SAEM 算法
│   ├── estep.py          # E 步
│   ├── mstep.py          # M 步
│   ├── conddist.py       # [新增] 条件分布估计
│   ├── likelihood.py     # 似然计算
│   └── ...
├── data.py               # 数据管理
├── model.py              # 模型定义
├── results.py            # 结果对象 [增强]
├── diagnostics.py        # 诊断功能 [增强]
├── compare.py            # [新增] 模型比较
├── stepwise.py           # [新增] 逐步回归
├── simulation.py         # [新增] 模拟功能
├── export.py             # [新增] 结果导出
└── plot_options.py       # [新增] 图形选项
```

## Components and Interfaces

### Component 1: 条件分布估计 (conddist.py)

```python
def conddist_saemix(
    saemix_object: SaemixObject,
    nsamp: int = 1,
    max_iter: Optional[int] = None,
    nburn: int = 0,
    nchains: int = 1,
    plot: bool = False,
    seed: Optional[int] = None
) -> SaemixObject:
    """
    使用 MCMC 算法估计个体参数的条件分布。
    
    实现 Metropolis-Hastings 采样，从 p(phi_i | y_i, theta) 中采样。
    
    Parameters
    ----------
    saemix_object : SaemixObject
        已拟合的 SAEM 结果对象
    nsamp : int
        每个个体的采样数量
    max_iter : int, optional
        最大 MCMC 迭代次数，默认为 nsamp * 10
    nburn : int
        burn-in 迭代次数
    nchains : int
        MCMC 链数量
    plot : bool
        是否显示收敛诊断图
    seed : int, optional
        随机数种子
    
    Returns
    -------
    SaemixObject
        更新后的结果对象，包含：
        - cond_mean_phi: 条件均值 (n_subjects, n_parameters)
        - cond_var_phi: 条件方差 (n_subjects, n_parameters)
        - cond_shrinkage: 收缩估计 (n_parameters,)
        - phi_samp: 采样结果 (n_subjects, nsamp, n_parameters)
    """
```

MCMC 采样算法：
1. 初始化：使用 MAP 估计作为起点
2. 提议分布：使用自适应高斯提议
3. 接受概率：基于后验密度比
4. 收敛诊断：Gelman-Rubin 统计量

### Component 2: 模型比较 (compare.py)

```python
def compare_saemix(
    *models: SaemixObject,
    method: str = 'is',
    names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    使用信息准则比较多个模型。
    
    Parameters
    ----------
    *models : SaemixObject
        多个已拟合的模型对象
    method : str
        似然计算方法: 'is' (重要性采样), 'gq' (高斯积分)
    names : list of str, optional
        模型名称列表
    
    Returns
    -------
    pd.DataFrame
        比较结果表，包含列：
        - model: 模型名称
        - npar: 参数数量
        - ll: 对数似然
        - AIC: Akaike 信息准则
        - BIC: Bayesian 信息准则
        - BIC_cov: 协变量选择专用 BIC
    """
```

信息准则计算：
- AIC = -2 * LL + 2 * k
- BIC = -2 * LL + log(N) * k
- BIC_cov = -2 * LL + log(n_total_obs) * k

### Component 3: 结果对象增强 (results.py)

```python
class SaemixRes:
    """增强的结果类"""
    
    # 现有属性
    fixed_effects: np.ndarray
    omega: np.ndarray
    respar: np.ndarray
    ll: float
    aic: float
    bic: float
    
    # 新增属性
    conf_int: pd.DataFrame          # 置信区间
    parpop: np.ndarray              # 每次迭代的总体参数
    allpar: np.ndarray              # 每次迭代的所有参数
    predictions: pd.DataFrame       # 预测值 DataFrame
    ires: np.ndarray                # 个体残差
    wres: np.ndarray                # 加权残差
    pd_: np.ndarray                 # 预测偏差
    se_fixed: np.ndarray            # 固定效应标准误
    
    def compute_confidence_intervals(
        self,
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """计算参数置信区间"""
    
    def compute_residuals(
        self,
        saemix_object: 'SaemixObject'
    ) -> None:
        """计算各类残差"""
```

### Component 4: 逐步回归 (stepwise.py)

```python
def forward_procedure(
    saemix_object: SaemixObject,
    covariates: Optional[List[str]] = None,
    parameters: Optional[List[str]] = None,
    trace: bool = True,
    criterion: str = 'BIC'
) -> SaemixObject:
    """
    前向选择协变量。
    
    从空模型开始，逐步添加能够最大程度改善 BIC 的协变量。
    """

def backward_procedure(
    saemix_object: SaemixObject,
    trace: bool = True,
    criterion: str = 'BIC'
) -> SaemixObject:
    """
    后向消除协变量。
    
    从完整模型开始，逐步移除对 BIC 贡献最小的协变量。
    """

def stepwise_procedure(
    saemix_object: SaemixObject,
    direction: str = 'both',
    covariate_init: Optional[Dict] = None,
    trace: bool = True,
    criterion: str = 'BIC'
) -> SaemixObject:
    """
    双向逐步回归。
    
    交替进行前向选择和后向消除。
    """
```

### Component 5: 模拟功能 (simulation.py)

```python
def simulate_saemix(
    saemix_object: SaemixObject,
    nsim: int = 1000,
    seed: Optional[int] = None,
    predictions: bool = True,
    res_var: bool = True
) -> pd.DataFrame:
    """
    从拟合模型模拟数据。
    
    Returns
    -------
    pd.DataFrame
        模拟数据，包含列：
        - sim: 模拟编号
        - id: 个体 ID
        - time: 时间
        - ysim: 模拟观测值
        - ppred: 群体预测 (if predictions=True)
        - ipred: 个体预测 (if predictions=True)
    """

def simulate_discrete_saemix(
    saemix_object: SaemixObject,
    simulate_function: Callable,
    nsim: int = 1000,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    离散响应模型模拟。
    """
```

### Component 6: 诊断图增强 (diagnostics.py)

```python
def plot_convergence(
    saemix_object: SaemixObject,
    parameters: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """绘制参数收敛图"""

def plot_likelihood(
    saemix_object: SaemixObject,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """绘制似然轨迹图"""

def plot_parameters_vs_covariates(
    saemix_object: SaemixObject,
    covariates: Optional[List[str]] = None,
    parameters: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """参数 vs 协变量图"""

def plot_randeff_vs_covariates(
    saemix_object: SaemixObject,
    covariates: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """随机效应 vs 协变量图"""

def plot_marginal_distribution(
    saemix_object: SaemixObject,
    parameters: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """参数边际分布图"""

def plot_correlations(
    saemix_object: SaemixObject,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """随机效应相关性图"""
```

### Component 7: 结果导出 (export.py)

```python
def save_results(
    saemix_object: SaemixObject,
    directory: str = 'results',
    overwrite: bool = True
) -> None:
    """
    保存所有结果到目录。
    
    创建文件：
    - parameters.csv: 参数估计
    - predictions.csv: 预测值
    - diagnostics.csv: 诊断统计
    - summary.txt: 结果摘要
    """

def export_to_csv(
    saemix_object: SaemixObject,
    filename: str,
    what: str = 'parameters'
) -> None:
    """
    导出指定内容到 CSV。
    
    what: 'parameters', 'predictions', 'residuals', 'eta'
    """

def save_plots(
    saemix_object: SaemixObject,
    directory: str = 'plots',
    format: str = 'png',
    dpi: int = 150
) -> None:
    """保存所有诊断图"""
```

### Component 8: 图形选项 (plot_options.py)

```python
@dataclass
class PlotOptions:
    """图形选项配置类"""
    figsize: Tuple[int, int] = (10, 8)
    dpi: int = 100
    style: str = 'seaborn-v0_8-whitegrid'
    color_palette: str = 'tab10'
    alpha: float = 0.7
    marker_size: int = 20
    line_width: float = 1.5
    font_size: int = 12
    title_size: int = 14
    
    def apply(self) -> None:
        """应用当前选项到 matplotlib"""

# 全局选项实例
_plot_options = PlotOptions()

def set_plot_options(**kwargs) -> None:
    """设置全局图形选项"""

def get_plot_options() -> PlotOptions:
    """获取当前图形选项"""

def reset_plot_options() -> None:
    """重置为默认选项"""
```

## Data Models

### 迭代历史记录

```python
@dataclass
class IterationHistory:
    """SAEM 迭代历史"""
    iteration: np.ndarray           # 迭代编号
    fixed_effects: np.ndarray       # (n_iter, n_fixed)
    omega_diag: np.ndarray          # (n_iter, n_random)
    respar: np.ndarray              # (n_iter, n_respar)
    ll: Optional[np.ndarray]        # (n_iter,) 似然值（如果计算）
```

### 预测结果 DataFrame

```python
predictions_schema = {
    'id': int,              # 个体 ID
    'time': float,          # 时间
    'yobs': float,          # 观测值
    'ppred': float,         # 群体预测
    'ipred': float,         # 个体预测 (MAP)
    'icpred': float,        # 个体预测 (条件均值)
    'ires': float,          # 个体残差
    'wres': float,          # 加权残差
    'iwres': float,         # 个体加权残差
    'npde': float           # NPDE
}
```

### 置信区间 DataFrame

```python
conf_int_schema = {
    'parameter': str,       # 参数名称
    'estimate': float,      # 估计值
    'se': float,            # 标准误
    'lower': float,         # 下界
    'upper': float,         # 上界
    'rse': float            # 相对标准误 (%)
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: MCMC Conditional Distribution Output Structure

*For any* fitted SaemixObject with n_subjects and n_parameters, when conddist_saemix is called, the returned object SHALL have:
- cond_mean_phi with shape (n_subjects, n_parameters)
- cond_var_phi with shape (n_subjects, n_parameters) with all values >= 0
- cond_shrinkage with shape (n_parameters,)
- phi_samp with shape (n_subjects, nsamp, n_parameters)

**Validates: Requirements 1.1, 1.2, 1.7**

### Property 2: Shrinkage Bounds

*For any* conditional distribution estimation result, all shrinkage values SHALL be between 0 and 1 (inclusive), where shrinkage = 1 - var(cond_mean) / var(population).

**Validates: Requirements 1.3**

### Property 3: Sample Count Consistency

*For any* valid nsamp parameter value, the phi_samp array SHALL have exactly nsamp samples per subject, i.e., phi_samp.shape[1] == nsamp.

**Validates: Requirements 1.4**

### Property 4: Model Comparison Output Correctness

*For any* set of fitted SaemixObjects, compare_saemix SHALL return a DataFrame where:
- AIC = -2 * ll + 2 * npar for each model
- BIC = -2 * ll + log(n_subjects) * npar for each model
- BIC_cov = -2 * ll + log(n_total_obs) * npar for each model
- All columns (model, npar, ll, AIC, BIC, BIC_cov) are present

**Validates: Requirements 2.1, 2.2, 2.3, 2.5**

### Property 5: Confidence Interval Computation

*For any* parameter estimate with standard error se, the 95% confidence interval SHALL be:
- lower = estimate - 1.96 * se
- upper = estimate + 1.96 * se
- The interval is symmetric around the estimate

**Validates: Requirements 3.1, 3.6**

### Property 6: Iteration History Recording

*For any* SAEM run with K iterations, the results SHALL contain:
- parpop with shape (K, n_fixed_effects)
- allpar with shape (K, n_total_parameters)
- Both arrays record values at each iteration

**Validates: Requirements 3.2, 3.3**

### Property 7: Predictions and Residuals Structure

*For any* fitted SaemixObject, the predictions DataFrame SHALL contain columns: id, time, yobs, ppred, ipred, and the residuals SHALL satisfy:
- ires = yobs - ipred
- wres = ires / g (where g is the error model function)

**Validates: Requirements 3.4, 3.5**

### Property 8: Stepwise Selection Optimality

*For any* stepwise selection result, the final model SHALL be locally optimal, meaning:
- For forward: no single covariate addition improves BIC
- For backward: no single covariate removal improves BIC
- For both: neither addition nor removal improves BIC

**Validates: Requirements 4.1, 4.2, 4.3, 4.5**

### Property 9: Simulation Reproducibility

*For any* seed value, calling simulate_saemix twice with the same seed SHALL produce identical results. This is a round-trip property: simulate(seed=s) == simulate(seed=s).

**Validates: Requirements 5.3**

### Property 10: Simulation Output Structure

*For any* simulation with nsim replicates, the output DataFrame SHALL:
- Have exactly nsim * n_observations rows
- Contain columns: sim, id, time, ysim
- When predictions=True, also contain ppred and ipred columns
- When res_var=True, ysim differs from ipred

**Validates: Requirements 5.1, 5.2, 5.4, 5.5, 5.7**

### Property 11: File Export Round-Trip

*For any* SaemixObject, saving results and then reading the saved files SHALL produce data equivalent to the original:
- export_to_csv(obj, 'params.csv', 'parameters'); read_csv('params.csv') contains all parameter estimates
- The directory is created if it doesn't exist

**Validates: Requirements 7.1, 7.2, 7.3, 7.4**

### Property 12: Plot Options Management

*For any* PlotOptions configuration:
- set_plot_options(figsize=(w, h)) followed by get_plot_options().figsize SHALL return (w, h)
- reset_plot_options() SHALL restore all options to their default values
- Local options override global options when both are specified

**Validates: Requirements 8.2, 8.3, 8.4, 8.5**

## Error Handling

### Conditional Distribution Errors
- Raise `ValueError` if SaemixObject has not been fitted (no results)
- Raise `ValueError` if nsamp < 1
- Raise `ValueError` if max_iter < nsamp

### Model Comparison Errors
- Raise `ValueError` if fewer than 2 models are provided
- Raise `ValueError` if models have different data (different n_subjects or n_observations)
- Raise `ValueError` if method is not one of 'is', 'gq', 'lin'

### Stepwise Regression Errors
- Raise `ValueError` if no covariates are available for selection
- Raise `ValueError` if direction is not one of 'forward', 'backward', 'both'
- Raise `RuntimeError` if SAEM fails to converge during model fitting

### Simulation Errors
- Raise `ValueError` if nsim < 1
- Raise `ValueError` if SaemixObject has not been fitted

### Export Errors
- Raise `ValueError` if what parameter is not recognized
- Raise `FileExistsError` if file exists and overwrite=False
- Raise `PermissionError` if directory cannot be created

## Testing Strategy

### Unit Tests
- Test each function with specific examples
- Test edge cases (empty data, single subject, single parameter)
- Test error conditions raise appropriate exceptions

### Property-Based Tests
Property-based testing will be implemented using `hypothesis` library.

Each property test will:
- Generate random valid inputs using hypothesis strategies
- Run minimum 100 iterations per property
- Tag tests with property references

**Test Configuration:**
```python
from hypothesis import given, settings, strategies as st

@settings(max_examples=100)
@given(...)
def test_property_N_description():
    """
    Feature: saemix-python-enhancement, Property N: Property Title
    Validates: Requirements X.Y
    """
```

### Integration Tests
- Test complete workflows from data loading to result export
- Test with real datasets from saemix-main/data/
- Compare results with R version outputs where possible

### Test Data
- Use `theo.saemix.tab` for PK model testing
- Use `cow.saemix.tab` for growth model testing
- Use synthetic data for edge case testing

