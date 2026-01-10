# Design Document: SAEMIX Python 鲁棒性优化

## Overview

本设计文档描述了 Python 版本 saemix 库的鲁棒性与工程化优化实现方案。主要目标是提升代码的正确性、可复现性、数值稳定性和工程化质量。

设计遵循以下原则：
- 不破坏现有 API，保持向后兼容
- 所有影响拟合结果的改动必须配套回归测试
- 优先修复高风险问题（静默失败、数据丢失）
- 使用 numpy.random.Generator 替代全局随机数

## Architecture

```
saemix/
├── data.py               # [修改] 辅助列处理、协变量校验修复
├── control.py            # [修改] RNG 统一入口
├── utils.py              # [修改] 参数变换数值防护、ID 转换助手
├── algorithm/
│   ├── saem.py           # [修改] RNG 传递
│   ├── estep.py          # [修改] RNG 使用
│   ├── mstep.py          # [修改] 数值稳定性
│   ├── conddist.py       # [修改] RNG 使用
│   └── ...
├── simulation.py         # [修改] RNG 使用
├── __init__.py           # [修改] 错误暴露
└── ...

.github/
└── workflows/
    └── ci.yml            # [新增] CI 工作流

tests/
├── test_data_robustness.py      # [新增] 数据处理鲁棒性测试
├── test_rng_isolation.py        # [新增] RNG 隔离测试
├── test_numerical_stability.py  # [新增] 数值稳定性测试
└── test_regression.py           # [新增] 回归测试
```

## Components and Interfaces

### Component 1: SaemixData 辅助列处理修复

**问题分析：**
当前 `_process_data` 方法在数据裁剪时，只保留了 `all_cols` 中的列，但 `mdv/cens/occ/ytype` 对应的原始列名可能不在 `all_cols` 中，导致这些列被丢弃。

**修复方案：**

```python
def _process_data(self):
    """处理数据，修复辅助列映射问题"""
    # 构建需要保留的列列表
    all_cols = [self.name_group] + self.name_predictors + [self.name_response]
    if self.name_covariates:
        all_cols.extend(self.name_covariates)
    
    # 新增：保留辅助列的原始列名
    auxiliary_cols = []
    auxiliary_mapping = {
        'mdv': self.name_mdv,
        'cens': self.name_cens,
        'occ': self.name_occ,
        'ytype': self.name_ytype
    }
    
    for internal_name, user_col in auxiliary_mapping.items():
        if user_col and user_col in self.data.columns:
            if user_col not in all_cols:
                auxiliary_cols.append(user_col)
            if self.verbose:
                print(f"Using column '{user_col}' for {internal_name}")
        elif user_col and user_col not in self.data.columns:
            raise ValueError(f"Specified {internal_name} column '{user_col}' not found in data")
        elif self.verbose:
            print(f"Using default values for {internal_name}")
    
    # 保留所有需要的列
    all_cols.extend(auxiliary_cols)
    self.data = self.data[all_cols].copy()
    
    # ... 后续处理保持不变，但使用原始列名映射
```

### Component 2: 协变量校验逻辑修复

**问题分析：**
当前代码在遍历 `name_covariates` 时调用 `remove()`，这会导致跳过某些元素。

**修复方案：**

```python
def _validate_data(self):
    """验证数据，修复协变量校验逻辑"""
    # ... 其他验证代码 ...
    
    # 修复：使用列表推导式构造新列表，而不是在遍历时修改
    if self.name_covariates:
        valid_covariates = []
        ignored_covariates = []
        
        for cov in self.name_covariates:
            if cov in self.data.columns:
                valid_covariates.append(cov)
            else:
                ignored_covariates.append(cov)
        
        if ignored_covariates and self.verbose:
            print(f"Warning: Covariate columns not found, ignoring: {ignored_covariates}")
        
        if not valid_covariates and self.verbose:
            print("Warning: No valid covariates found, covariate list is empty")
        
        self.name_covariates = valid_covariates
```

### Component 3: 统一随机数管理

**设计方案：**

1. 在 `saemix_control` 中创建 `numpy.random.Generator` 实例
2. 通过参数传递 RNG 到所有需要随机数的模块
3. 不再调用 `np.random.seed()` 或使用全局随机状态

```python
# control.py
from typing import Optional
import numpy as np

def saemix_control(
    # ... 现有参数 ...
    seed: Optional[int] = 23456,
    rng: Optional[np.random.Generator] = None,
    # ...
) -> Dict[str, Any]:
    """
    创建 SAEM 控制参数。
    
    Parameters
    ----------
    seed : int, optional
        随机数种子，用于创建 RNG
    rng : numpy.random.Generator, optional
        用户提供的 RNG 实例，如果提供则忽略 seed
    """
    # 创建 RNG 实例
    if rng is not None:
        _rng = rng
    elif seed is not None:
        _rng = np.random.default_rng(seed)
    else:
        _rng = np.random.default_rng()
    
    control = {
        # ... 现有参数 ...
        'rng': _rng,
        'seed': seed,
        # ...
    }
    return control
```

**RNG 传递模式：**

```python
# simulation.py
def simulate_saemix(
    saemix_object: SaemixObject,
    nsim: int = 1000,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    # ...
) -> pd.DataFrame:
    """
    模拟数据。
    
    Parameters
    ----------
    seed : int, optional
        随机数种子（向后兼容）
    rng : numpy.random.Generator, optional
        RNG 实例，优先于 seed
    """
    # 确定使用的 RNG
    if rng is not None:
        _rng = rng
    elif seed is not None:
        _rng = np.random.default_rng(seed)
    else:
        # 从 saemix_object 获取 RNG，如果没有则创建新的
        _rng = getattr(saemix_object, 'rng', None) or np.random.default_rng()
    
    # 使用 _rng 进行所有随机操作
    # 替换 np.random.xxx() 为 _rng.xxx()
    eta = _rng.multivariate_normal(mean, cov, size=n)
    epsilon = _rng.standard_normal(n_obs)
    # ...
```

### Component 4: 参数变换数值防护

**修复方案：**

```python
# utils.py
import warnings
from typing import Optional

# 数值常量
LOG_EPS = 1e-10
LOGIT_EPS = 1e-10

def transphi(phi, tr, verbose: bool = False):
    """
    将 phi (未变换的参数) 转换为 psi (变换后的参数)
    
    增加数值防护，防止 NaN/Inf
    """
    psi = phi.copy()
    if phi.ndim == 1:
        psi = psi.reshape(1, -1)
        was_1d = True
    else:
        was_1d = False
    
    tr = np.array(tr)
    
    # Log 变换 (tr == 1)
    i1 = np.where(tr == 1)[0]
    if len(i1) > 0:
        # 无需 clip，因为 exp 对任何有限输入都产生有限输出
        psi[:, i1] = np.exp(psi[:, i1])
        # 检查溢出
        if np.any(np.isinf(psi[:, i1])):
            raise ValueError(
                f"Log transformation overflow: input values too large. "
                f"Max input: {np.max(phi[:, i1])}"
            )
    
    # Probit 变换 (tr == 2)
    i2 = np.where(tr == 2)[0]
    if len(i2) > 0:
        psi[:, i2] = norm.cdf(psi[:, i2])
    
    # Logit 变换 (tr == 3)
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
        clipped = np.clip(phi[:, i1], LOG_EPS, None)
        if verbose and np.any(phi[:, i1] < LOG_EPS):
            n_clipped = np.sum(phi[:, i1] < LOG_EPS)
            warnings.warn(
                f"Log inverse transform: {n_clipped} values clipped to {LOG_EPS}"
            )
        phi[:, i1] = np.log(clipped)
    
    # Probit 逆变换 (tr == 2)
    i2 = np.where(tr == 2)[0]
    if len(i2) > 0:
        clipped = np.clip(phi[:, i2], LOGIT_EPS, 1 - LOGIT_EPS)
        if verbose and np.any((phi[:, i2] < LOGIT_EPS) | (phi[:, i2] > 1 - LOGIT_EPS)):
            warnings.warn("Probit inverse transform: values clipped to (eps, 1-eps)")
        phi[:, i2] = norm.ppf(clipped)
    
    # Logit 逆变换 (tr == 3): phi = log(psi / (1 - psi))
    i3 = np.where(tr == 3)[0]
    if len(i3) > 0:
        # Clip 防止 log(0) 和 log(inf)
        clipped = np.clip(phi[:, i3], LOGIT_EPS, 1 - LOGIT_EPS)
        if verbose and np.any((phi[:, i3] < LOGIT_EPS) | (phi[:, i3] > 1 - LOGIT_EPS)):
            warnings.warn(
                f"Logit inverse transform: values clipped to ({LOGIT_EPS}, {1-LOGIT_EPS})"
            )
        phi[:, i3] = np.log(clipped / (1 - clipped))
    
    # 最终检查
    if np.any(~np.isfinite(phi)):
        bad_indices = np.where(~np.isfinite(phi))
        raise ValueError(
            f"Transformation produced non-finite values at indices {bad_indices}. "
            f"Input range: [{np.nanmin(psi)}, {np.nanmax(psi)}]"
        )
    
    if was_1d:
        phi = phi.flatten()
    
    return phi
```

### Component 5: ID 转换助手函数

```python
# utils.py

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
    """
    if index < 0:
        raise ValueError(f"Index must be >= 0, got {index}")
    return index + 1
```

### Component 6: 数值稳定性增强

**M-step 协方差矩阵计算：**

```python
# algorithm/mstep.py

def compute_omega_safe(phi_samples: np.ndarray, mu: np.ndarray, 
                       min_eigenvalue: float = 1e-8) -> np.ndarray:
    """
    安全计算协方差矩阵，确保正定性。
    
    Parameters
    ----------
    phi_samples : np.ndarray
        个体参数样本，形状 (n_subjects, n_params)
    mu : np.ndarray
        均值向量
    min_eigenvalue : float
        最小特征值阈值
    
    Returns
    -------
    np.ndarray
        正定协方差矩阵
    
    Raises
    ------
    ValueError
        如果无法修正为正定矩阵
    """
    # 计算样本协方差
    centered = phi_samples - mu
    omega = np.dot(centered.T, centered) / len(phi_samples)
    
    # 检查并修正正定性
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(omega)
        
        if np.any(eigenvalues < min_eigenvalue):
            warnings.warn(
                f"Covariance matrix has small/negative eigenvalues: "
                f"min={eigenvalues.min():.2e}. Correcting to ensure positive definiteness."
            )
            # 将小于阈值的特征值设为阈值
            eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
            omega = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            # 确保对称性
            omega = (omega + omega.T) / 2
        
        # 验证 Cholesky 分解可行
        np.linalg.cholesky(omega)
        
    except np.linalg.LinAlgError as e:
        raise ValueError(
            f"Failed to compute valid covariance matrix. "
            f"Original error: {e}. "
            f"Check model specification and data quality."
        )
    
    return omega
```

**对数似然计算防护：**

```python
# algorithm/likelihood.py

def compute_log_likelihood_safe(
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    sigma: float,
    log_eps: float = -700  # 防止 exp 下溢
) -> float:
    """
    安全计算对数似然。
    
    Parameters
    ----------
    y_obs : np.ndarray
        观测值
    y_pred : np.ndarray
        预测值
    sigma : float
        残差标准差
    log_eps : float
        对数似然下界
    
    Returns
    -------
    float
        对数似然值
    
    Raises
    ------
    ValueError
        如果计算结果为 NaN 或 +Inf
    """
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")
    
    residuals = y_obs - y_pred
    n = len(y_obs)
    
    # 计算对数似然
    ll = -0.5 * n * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum(residuals**2) / sigma**2
    
    # 检查结果
    if np.isnan(ll):
        raise ValueError(
            f"Log-likelihood is NaN. "
            f"y_obs range: [{y_obs.min():.4g}, {y_obs.max():.4g}], "
            f"y_pred range: [{y_pred.min():.4g}, {y_pred.max():.4g}], "
            f"sigma: {sigma:.4g}"
        )
    
    if np.isinf(ll) and ll > 0:
        raise ValueError(
            f"Log-likelihood is +Inf, which is invalid. "
            f"Check for zero residuals or invalid sigma."
        )
    
    # 下界截断（防止数值问题）
    ll = max(ll, log_eps)
    
    return ll
```

### Component 7: CI 工作流配置

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest tests/ -v --tb=short
    
    - name: Build verification
      run: |
        python -m build
        pip install dist/*.whl --force-reinstall
        python -c "import saemix; print(saemix.__version__)"

  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    - name: Lint
      run: |
        pip install ruff
        ruff check saemix/
```

### Component 8: 错误处理与依赖管理

```python
# __init__.py

import warnings

# 核心依赖（必须）
try:
    import numpy as np
    import pandas as pd
    from scipy import stats
except ImportError as e:
    raise ImportError(
        f"Missing required dependency: {e.name}. "
        f"Please install with: pip install saemix"
    ) from e

# 可选依赖
_HAS_MATPLOTLIB = False
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    pass

def _require_matplotlib():
    """检查 matplotlib 是否可用，不可用时抛出清晰错误"""
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting functionality. "
            "Install with: pip install matplotlib"
        )

# 导出公共 API
from .data import SaemixData
from .model import SaemixModel
from .control import saemix_control
from .main import saemix
from .results import SaemixResults
from .simulation import simulate_saemix
from .diagnostics import compute_diagnostics
from .export import export_results
from .compare import compare_models
from .stepwise import stepwise_covariate

# 版本信息
from ._version import __version__

__all__ = [
    'SaemixData',
    'SaemixModel', 
    'saemix_control',
    'saemix',
    'SaemixResults',
    'simulate_saemix',
    'compute_diagnostics',
    'export_results',
    'compare_models',
    'stepwise_covariate',
    '__version__',
]
```


## Data Models

### RNG 配置模型

```python
@dataclass
class RNGConfig:
    """随机数生成器配置"""
    seed: Optional[int] = 23456
    rng: Optional[np.random.Generator] = None
    
    def get_rng(self) -> np.random.Generator:
        """获取 RNG 实例"""
        if self.rng is not None:
            return self.rng
        elif self.seed is not None:
            return np.random.default_rng(self.seed)
        else:
            return np.random.default_rng()
```

### 辅助列配置模型

```python
@dataclass
class AuxiliaryColumns:
    """辅助列配置"""
    mdv: Optional[str] = None      # Missing Dependent Variable
    cens: Optional[str] = None     # Censoring indicator
    occ: Optional[str] = None      # Occasion
    ytype: Optional[str] = None    # Response type
    
    def get_column_mapping(self) -> Dict[str, Optional[str]]:
        """获取列名映射"""
        return {
            'mdv': self.mdv,
            'cens': self.cens,
            'occ': self.occ,
            'ytype': self.ytype
        }
    
    def get_required_columns(self) -> List[str]:
        """获取需要保留的列名列表"""
        return [col for col in [self.mdv, self.cens, self.occ, self.ytype] 
                if col is not None]
```

### 数值防护配置

```python
@dataclass
class NumericalConfig:
    """数值计算配置"""
    log_eps: float = 1e-10           # log 变换最小值
    logit_eps: float = 1e-10         # logit 变换边界
    min_eigenvalue: float = 1e-8     # 协方差矩阵最小特征值
    ll_lower_bound: float = -700     # 对数似然下界
    verbose: bool = False            # 是否输出警告
```

### 回归测试参考值模型

```python
@dataclass
class RegressionReference:
    """回归测试参考值"""
    dataset_name: str
    seed: int
    fixed_effects: Dict[str, float]
    omega: np.ndarray
    sigma: float
    log_likelihood: float
    aic: float
    bic: float
    tolerance: Dict[str, float]  # 各指标的容差
    
    def check_results(self, results: 'SaemixResults') -> Dict[str, bool]:
        """检查结果是否在容差范围内"""
        checks = {}
        # 检查固定效应
        for param, ref_value in self.fixed_effects.items():
            actual = results.fixed_effects.get(param)
            tol = self.tolerance.get('fixed_effects', 0.01)
            checks[f'fixed_{param}'] = abs(actual - ref_value) / abs(ref_value) < tol
        # 检查其他指标...
        return checks
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Auxiliary Column Mapping Preservation

*For any* valid DataFrame with auxiliary columns (mdv, cens, occ, ytype) and corresponding column name specifications, when SaemixData processes the data, the auxiliary columns SHALL be correctly mapped and accessible after all internal transformations.

**Validates: Requirements 1.1**

### Property 2: Auxiliary Column Error Handling

*For any* column name specification where the specified column does not exist in the DataFrame, SaemixData SHALL raise a ValueError containing the missing column name.

**Validates: Requirements 1.2**

### Property 3: Auxiliary Column Data Integrity

*For any* valid auxiliary column data, after SaemixData completes all internal transformations (subsetting, filtering, sorting), the auxiliary column values SHALL be equivalent to the original values for the corresponding rows.

**Validates: Requirements 1.5**

### Property 4: Covariate Validation Correctness

*For any* list of covariate names where some exist in the DataFrame and some do not, after validation the covariate list SHALL contain exactly the covariates that exist in the DataFrame, regardless of their original position in the input list.

**Validates: Requirements 2.1, 2.2**

### Property 5: RNG Reproducibility

*For any* valid input data, model, and seed value, running the SAEM algorithm twice with identical inputs SHALL produce identical results within numerical precision (relative tolerance < 1e-10).

**Validates: Requirements 3.5**

### Property 6: Global RNG State Preservation

*For any* global numpy random state set before calling saemix functions, the global state SHALL be unchanged after the function completes execution.

**Validates: Requirements 3.3, 3.4, 3.6**

### Property 7: Transformation Output Finiteness

*For any* input array and transformation type (log, logit, probit), the transformation functions SHALL produce finite output values (no NaN, no Inf, no -Inf) for all inputs within the valid domain, and SHALL clip boundary values to ensure finiteness.

**Validates: Requirements 4.1, 4.2**

### Property 8: Transformation Error Handling

*For any* input that would cause numerical overflow (e.g., extremely large values for exp), the transformation functions SHALL raise a ValueError with diagnostic information including the input range.

**Validates: Requirements 4.3**

### Property 9: Inverse Transformation Bounds

*For any* valid transformed parameter values, the inverse transformation SHALL produce finite results by applying appropriate bounds before computation.

**Validates: Requirements 4.5**

### Property 10: Singular Matrix Error Handling

*For any* singular or near-singular matrix input to matrix operations, the algorithm SHALL raise a descriptive error indicating the operation type and possible causes.

**Validates: Requirements 5.1**

### Property 11: Log-Likelihood Error Handling

*For any* input combination that would produce NaN or +Inf log-likelihood, the algorithm SHALL raise a ValueError with diagnostic information.

**Validates: Requirements 5.2**

### Property 12: Covariance Matrix Correction

*For any* computed covariance matrix with small negative eigenvalues (due to numerical precision), the algorithm SHALL correct it to be positive definite while issuing a warning.

**Validates: Requirements 5.4**

### Property 13: ID Conversion Round-Trip

*For any* valid user-facing 1-based ID, converting to internal 0-based index and back SHALL produce the original ID. Similarly, for any valid 0-based index, converting to 1-based ID and back SHALL produce the original index.

**Validates: Requirements 7.4**


## Error Handling

### Error Categories

| Category | Error Type | Handling Strategy |
|----------|-----------|-------------------|
| 数据验证 | ValueError | 立即抛出，包含缺失列名或无效值信息 |
| 数值计算 | ValueError | 抛出时包含输入范围和迭代信息 |
| 矩阵运算 | LinAlgError | 捕获并转换为描述性 ValueError |
| 依赖缺失 | ImportError | 核心依赖立即抛出，可选依赖延迟到使用时 |
| 配置错误 | ValueError | 在初始化时验证并抛出 |

### Error Message Format

所有错误消息应遵循以下格式：

```
[Component] Error description.
Context: relevant_variable=value, another_variable=value
Suggestion: Possible solution or next step.
```

示例：

```python
raise ValueError(
    "[SaemixData] Specified mdv column 'MDV_COL' not found in data. "
    f"Context: available_columns={list(df.columns)[:10]}... "
    "Suggestion: Check column name spelling or use name_mdv=None for default."
)
```

### Warning Categories

| Category | Warning Type | Trigger Condition |
|----------|-------------|-------------------|
| 数值截断 | UserWarning | 参数变换时值被 clip |
| 协变量缺失 | UserWarning | 指定的协变量列不存在 |
| 矩阵修正 | UserWarning | 协方差矩阵特征值被修正 |
| 收敛问题 | ConvergenceWarning | 算法未在最大迭代内收敛 |

## Testing Strategy

### 测试框架

- **单元测试**: pytest
- **属性测试**: hypothesis
- **覆盖率**: pytest-cov (目标 > 80%)

### 测试类型

#### 1. 属性测试 (Property-Based Tests)

使用 hypothesis 库实现，每个属性测试至少运行 100 次迭代。

```python
from hypothesis import given, strategies as st, settings
import numpy as np

# 示例：ID 转换往返测试
@given(st.integers(min_value=1, max_value=10000))
@settings(max_examples=100)
def test_id_conversion_round_trip(user_id):
    """
    Feature: saemix-robustness-optimization
    Property 13: ID Conversion Round-Trip
    Validates: Requirements 7.4
    """
    from saemix.utils import id_to_index, index_to_id
    
    index = id_to_index(user_id)
    recovered_id = index_to_id(index)
    assert recovered_id == user_id
```

#### 2. 单元测试 (Unit Tests)

针对特定示例和边界情况：

```python
def test_auxiliary_column_missing_raises_error():
    """测试缺失辅助列时抛出正确错误"""
    df = pd.DataFrame({'ID': [1, 2], 'TIME': [0, 1], 'DV': [1.0, 2.0]})
    
    with pytest.raises(ValueError, match="MDV_COL.*not found"):
        SaemixData(df, name_group='ID', name_predictors=['TIME'],
                   name_response='DV', name_mdv='MDV_COL')
```

#### 3. 回归测试 (Regression Tests)

使用固定数据集和种子验证输出稳定性：

```python
def test_theo_regression():
    """
    回归测试：theo 数据集
    
    参考值来源：R saemix 包 v3.2 运行结果
    容差说明：固定效应 1%，随机效应 5%，似然 0.1%
    """
    # 加载参考数据
    data = load_theo_data()
    model = create_theo_model()
    control = saemix_control(seed=12345)
    
    results = saemix(data, model, control)
    
    # 验证固定效应
    assert_allclose(results.fixed_effects['ka'], 1.5, rtol=0.01)
    assert_allclose(results.fixed_effects['V'], 35.0, rtol=0.01)
    assert_allclose(results.fixed_effects['Cl'], 3.0, rtol=0.01)
    
    # 验证似然
    assert_allclose(results.log_likelihood, -120.5, rtol=0.001)
```

### 测试文件结构

```
tests/
├── test_data_robustness.py       # 数据处理鲁棒性 (Properties 1-4)
├── test_rng_isolation.py         # RNG 隔离 (Properties 5-6)
├── test_numerical_stability.py   # 数值稳定性 (Properties 7-12)
├── test_utils_properties.py      # 工具函数 (Property 13)
├── test_regression.py            # 回归测试
└── conftest.py                   # 共享 fixtures
```

### 属性测试配置

```python
# conftest.py
from hypothesis import settings, Verbosity

# 全局设置
settings.register_profile("ci", max_examples=100, deadline=None)
settings.register_profile("dev", max_examples=20, deadline=None)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)

# 根据环境选择配置
import os
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
```

### 测试数据生成策略

```python
# conftest.py
from hypothesis import strategies as st
import pandas as pd
import numpy as np

@st.composite
def valid_saemix_dataframe(draw):
    """生成有效的 SaemixData 输入 DataFrame"""
    n_subjects = draw(st.integers(min_value=2, max_value=20))
    n_obs_per_subject = draw(st.integers(min_value=3, max_value=10))
    
    ids = np.repeat(range(1, n_subjects + 1), n_obs_per_subject)
    times = np.tile(np.linspace(0, 24, n_obs_per_subject), n_subjects)
    dv = draw(st.lists(
        st.floats(min_value=0.1, max_value=100, allow_nan=False),
        min_size=len(ids), max_size=len(ids)
    ))
    
    df = pd.DataFrame({'ID': ids, 'TIME': times, 'DV': dv})
    
    # 可选：添加辅助列
    if draw(st.booleans()):
        df['MDV'] = draw(st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=len(ids), max_size=len(ids)
        ))
    
    return df

@st.composite
def transformation_inputs(draw, transform_type):
    """生成参数变换测试输入"""
    n = draw(st.integers(min_value=1, max_value=100))
    
    if transform_type == 'log':
        # log 变换的有效输入范围
        values = draw(st.lists(
            st.floats(min_value=1e-10, max_value=1e10, allow_nan=False),
            min_size=n, max_size=n
        ))
    elif transform_type == 'logit':
        # logit 变换的有效输入范围
        values = draw(st.lists(
            st.floats(min_value=1e-10, max_value=1-1e-10, allow_nan=False),
            min_size=n, max_size=n
        ))
    else:
        values = draw(st.lists(
            st.floats(min_value=-10, max_value=10, allow_nan=False),
            min_size=n, max_size=n
        ))
    
    return np.array(values)
```

