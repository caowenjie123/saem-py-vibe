# Python saemix 开发计划

本文档详细描述 R 版本 saemix-main 与 Python 版本 saemix 之间的功能差异，以及 Python 版本尚未实现的功能。

## 1. 功能对比总览

| 功能模块 | R 版本 | Python 版本 | 状态 |
|---------|--------|-------------|------|
| 核心 SAEM 算法 | ✅ | ✅ | 已实现 |
| 数据管理 (SaemixData) | ✅ | ✅ | 已实现 |
| 模型定义 (SaemixModel) | ✅ | ✅ | 已实现 |
| 结果对象 (SaemixObject/SaemixRes) | ✅ | ⚠️ | 部分实现 |
| MAP 估计 | ✅ | ✅ | 已实现 |
| Fisher 信息矩阵 | ✅ | ✅ | 已实现 |
| 似然计算 (IS/GQ/Lin) | ✅ | ⚠️ | 部分实现 |
| 条件分布估计 | ✅ | ❌ | 未实现 |
| 模型比较 (AIC/BIC) | ✅ | ❌ | 未实现 |
| 逐步回归 (Stepwise) | ✅ | ❌ | 未实现 |
| 模拟功能 | ✅ | ⚠️ | 部分实现 |
| NPDE 计算 | ✅ | ✅ | 已实现 |
| VPC 图 | ✅ | ✅ | 已实现 |
| 诊断图 | ✅ | ⚠️ | 部分实现 |
| 结果保存/导出 | ✅ | ❌ | 未实现 |

## 2. 详细功能差异

### 2.1 已实现功能

#### 核心算法 (saemix/algorithm/)
- ✅ SAEM 主算法 (`saem.py`)
- ✅ E 步 (`estep.py`)
- ✅ M 步 (`mstep.py`)
- ✅ 初始化 (`initialization.py`)
- ✅ MAP 估计 (`map_estimation.py`)
- ✅ Fisher 信息矩阵 (`fim.py`)
- ✅ 似然计算 (`likelihood.py`)
- ✅ 预测 (`predict.py`)

#### 数据管理 (saemix/data.py)
- ✅ 数据加载 (DataFrame/文件)
- ✅ 列名验证
- ✅ 个体 ID 映射
- ✅ 缺失数据处理 (mdv)
- ✅ 删失数据处理 (cens)
- ✅ 多响应类型 (ytype)
- ✅ 协变量处理

#### 模型定义 (saemix/model.py)
- ✅ 结构模型
- ✅ 似然模型
- ✅ 误差模型 (constant/proportional/combined/exponential)
- ✅ 参数变换 (normal/log-normal/probit/logit)
- ✅ 协变量模型
- ✅ 协方差模型

#### 诊断功能 (saemix/diagnostics.py)
- ✅ 观测值 vs 预测值图
- ✅ 残差图
- ✅ 个体拟合图
- ✅ GOF 综合图
- ✅ NPDE 计算和图
- ✅ VPC 图
- ✅ 随机效应分布图

### 2.2 部分实现功能

#### 结果对象 (saemix/results.py)
Python 版本的 `SaemixRes` 相比 R 版本缺少以下属性：
- ❌ `conf.int` - 置信区间 DataFrame
- ❌ `parpop` - 每次迭代的总体参数
- ❌ `allpar` - 每次迭代的所有参数
- ❌ `predictions` - 预测值 DataFrame
- ❌ `ires` - 个体残差
- ❌ `wres` - 总体加权残差
- ❌ `pd` - 预测偏差

#### 似然计算
- ✅ 重要性采样 (IS)
- ✅ 高斯积分 (GQ)
- ❌ 线性化方法 (Lin) - R 版本在 FIM 计算时同时计算

#### 诊断图
R 版本提供更丰富的图形选项：
- ❌ 收敛图 (`saemix.plot.convergence`)
- ❌ 似然估计图 (`saemix.plot.llis`)
- ❌ 参数 vs 协变量图 (`saemix.plot.parcov`)
- ❌ 随机效应 vs 协变量图 (`saemix.plot.randeffcov`)
- ❌ 参数边际分布图 (`saemix.plot.distpsi`)
- ❌ 随机效应相关性图 (`saemix.plot.correlations`)
- ❌ Mirror 图 (`saemix.plot.mirror`)

### 2.3 未实现功能

#### 条件分布估计 (`func_distcond.R` → `conddist.saemix`)
R 版本提供完整的条件分布估计功能：
- 使用 MCMC 算法估计个体参数的条件均值和方差
- 支持多链采样
- 收敛诊断
- 条件收缩估计

**优先级：高**

#### 模型比较 (`func_compare.R` → `compare.saemix`)
R 版本支持多模型比较：
- AIC 比较
- BIC 比较
- BIC.cov (协变量选择专用 BIC)
- 支持不同似然计算方法

**优先级：高**

#### 逐步回归 (`func_stepwise.R`, `forward.R`, `backward.R`, `stepwise.R`)
R 版本提供完整的逐步回归功能：
- 前向选择 (`forward.procedure`)
- 后向消除 (`backward.procedure`)
- 双向逐步 (`stepwise.procedure`)
- 基于 BIC 的协变量和随机效应联合选择

**优先级：中**

#### 模拟功能 (`func_simulations.R`)
R 版本提供更完整的模拟功能：
- `simulate.SaemixObject` - 从拟合模型模拟
- `simulateDiscreteSaemix` - 离散响应模型模拟
- 支持不确定性传播

**优先级：中**

#### 结果保存/导出
R 版本支持：
- 结果保存到文件
- 图形保存
- 输出目录管理

**优先级：低**

#### 图形选项系统 (`func_plots.R`)
R 版本有完整的图形选项系统：
- `saemix.plot.setoptions` - 设置默认选项
- `replace.plot.options` - 替换选项
- 丰富的自定义参数

**优先级：低**

## 3. 开发计划

### Phase 1: 核心功能完善 (高优先级)

#### 1.1 条件分布估计
**文件**: `saemix/algorithm/conddist.py`

```python
def conddist_saemix(saemix_object, nsamp=1, max_iter=None, plot=False):
    """
    使用 MCMC 算法估计个体参数的条件均值和方差
    
    参数
    -----
    saemix_object : SaemixObject
        拟合结果对象
    nsamp : int
        采样数量
    max_iter : int
        最大迭代次数
    plot : bool
        是否显示收敛图
    
    返回
    -----
    SaemixObject
        更新后的结果对象，包含：
        - cond_mean_phi: 条件均值
        - cond_var_phi: 条件方差
        - cond_shrinkage: 收缩估计
        - phi_samp: 采样结果
    """
```

**工作量估计**: 3-5 天

#### 1.2 模型比较
**文件**: `saemix/compare.py`

```python
def compare_saemix(*models, method='is'):
    """
    使用信息准则比较多个模型
    
    参数
    -----
    *models : SaemixObject
        多个拟合结果对象
    method : str
        似然计算方法 ('is', 'lin', 'gq')
    
    返回
    -----
    DataFrame
        包含 AIC, BIC, BIC.cov 的比较表
    """
```

**工作量估计**: 1-2 天

#### 1.3 完善结果对象
**文件**: `saemix/results.py`

需要添加：
- 置信区间计算
- 迭代历史记录
- 预测值 DataFrame
- 残差计算

**工作量估计**: 2-3 天

### Phase 2: 模型选择功能 (中优先级)

#### 2.1 逐步回归
**文件**: `saemix/stepwise.py`

```python
def step_saemix(saemix_object, direction='forward', trace=True):
    """
    逐步回归进行协变量和随机效应选择
    
    参数
    -----
    saemix_object : SaemixObject
        初始拟合结果
    direction : str
        'forward', 'backward', 'both'
    trace : bool
        是否打印过程
    
    返回
    -----
    SaemixObject
        最优模型的拟合结果
    """

def forward_procedure(saemix_object, trace=True):
    """前向选择"""

def backward_procedure(saemix_object, trace=True):
    """后向消除"""

def stepwise_procedure(saemix_object, covariate_init=None, trace=True):
    """双向逐步"""
```

**工作量估计**: 5-7 天

#### 2.2 完善模拟功能
**文件**: `saemix/simulation.py`

```python
def simulate_saemix(saemix_object, nsim=1000, seed=None, 
                    predictions=True, res_var=True):
    """
    从拟合模型模拟数据
    """

def simulate_discrete_saemix(saemix_object, simulate_function, 
                             nsim=1000, seed=None):
    """
    离散响应模型模拟
    """
```

**工作量估计**: 2-3 天

### Phase 3: 诊断功能增强 (中优先级)

#### 3.1 收敛诊断图
```python
def plot_convergence(saemix_object):
    """绘制参数收敛图"""

def plot_likelihood(saemix_object):
    """绘制似然估计图"""
```

#### 3.2 参数-协变量关系图
```python
def plot_parameters_vs_covariates(saemix_object):
    """参数 vs 协变量散点图/箱线图"""

def plot_randeff_vs_covariates(saemix_object):
    """随机效应 vs 协变量图"""
```

#### 3.3 参数分布图
```python
def plot_marginal_distribution(saemix_object):
    """参数边际分布图"""

def plot_correlations(saemix_object):
    """随机效应相关性图"""
```

**工作量估计**: 3-4 天

### Phase 4: 辅助功能 (低优先级)

#### 4.1 结果保存/导出
```python
def save_results(saemix_object, directory='results'):
    """保存结果到文件"""

def export_to_csv(saemix_object, filename):
    """导出结果为 CSV"""
```

#### 4.2 图形选项系统
```python
class PlotOptions:
    """图形选项管理类"""
    
def set_plot_options(**kwargs):
    """设置全局图形选项"""
```

**工作量估计**: 2-3 天

## 4. 时间线估计

| 阶段 | 功能 | 预计时间 | 累计时间 |
|------|------|----------|----------|
| Phase 1.1 | 条件分布估计 | 3-5 天 | 3-5 天 |
| Phase 1.2 | 模型比较 | 1-2 天 | 4-7 天 |
| Phase 1.3 | 完善结果对象 | 2-3 天 | 6-10 天 |
| Phase 2.1 | 逐步回归 | 5-7 天 | 11-17 天 |
| Phase 2.2 | 完善模拟功能 | 2-3 天 | 13-20 天 |
| Phase 3 | 诊断功能增强 | 3-4 天 | 16-24 天 |
| Phase 4 | 辅助功能 | 2-3 天 | 18-27 天 |

**总计**: 约 3-4 周

## 5. 测试计划

每个新功能需要：
1. 单元测试
2. 与 R 版本结果对比验证
3. 使用 `saemix-main/data/` 中的示例数据测试

### 测试数据集
- `theo.saemix.tab` - 茶碱 PK 数据
- `cow.saemix.tab` - 奶牛生长数据
- `PD1.saemix.tab` - PD 数据
- `lung.saemix.tab` - 肺功能数据

## 6. 注意事项

### 6.1 索引差异
Python 使用 0-based 索引，R 使用 1-based 索引。在移植代码时需要特别注意：
- 矩阵/数组索引
- 循环索引
- 参数位置索引

### 6.2 命名约定
- R: `camelCase` 或 `dot.case`
- Python: `snake_case`

### 6.3 数据结构
- R `matrix` → Python `numpy.ndarray`
- R `data.frame` → Python `pandas.DataFrame`
- R `list` → Python `dict` 或 `list`

## 7. 参考资源

- R saemix 源码: `saemix-main/R/`
- R saemix 文档: `saemix-main/man/`
- 示例数据: `saemix-main/data/`
- 设计文档: `DESIGN.md`
