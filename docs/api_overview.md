# 核心概念与 API 总览

本页是对 `saemix` 包对外 API 的“导航页”，帮助你从整体理解：

- **数据**如何组织
- **模型**如何表达
- **控制参数**如何设置
- **结果对象**能做什么

> 说明：更完整的参数解释可结合源码与示例脚本（`demo_estimation.py`、`examples/basic_example.py`）。

## 1. 数据：`saemix_data` / `SaemixData`

用于管理纵向数据，并完成必要的校验与预测变量矩阵构造。

典型入参：

- `name_data`
  - `pandas.DataFrame` 或 数据文件路径
- `name_group`
  - 个体 ID 列名（例如 `"Id"`）
- `name_predictors`
  - 预测变量列名列表（例如 `['Dose', 'Time']`）
- `name_response`
  - 响应变量列名（例如 `"Concentration"`）
- `name_covariates`（可选）
  - 协变量列名列表

示例：

```python
saemix_data_obj = saemix_data(
    name_data=data,
    name_group="Id",
    name_predictors=["Dose", "Time"],
    name_response="Concentration",
)
```

## 2. 模型：`saemix_model` / `SaemixModel`

用于定义 NLME 模型结构、参数初始值、随机效应结构、误差模型等。

### 2.1 模型函数签名

`model(psi, id, xidep)`：

- `psi`：个体参数矩阵（shape 通常为 `n_subjects x n_parameters`）
- `id`：当前个体的 0-based 索引（`0..N-1`）
- `xidep`：预测变量矩阵（shape 通常为 `n_obs_for_subject x n_predictors`）

你的模型函数应返回当前个体每条观测对应的预测值 `ypred`。

### 2.2 常用入参

- `psi0`
  - 初始总体参数（通常是 `1 x n_parameters` 的 `numpy.ndarray`）
- `name_modpar`
  - 参数名称列表（用于输出/对齐）
- `transform_par`
  - 参数变换类型列表
  - 常见约定：`0=normal`，`1=log-normal`（其余如 probit/logit 视实现而定）
- `covariance_model`
  - 随机效应协方差结构（例如 `np.eye(p)` 表示全随机效应；对角/稀疏结构用于固定部分随机效应）
- `omega_init`
  - Omega 初值（随机效应协方差矩阵）
- `error_model`
  - 误差模型字符串，例如：`constant`、`proportional`、`combined`、`exponential`
- `error_init`（示例中出现）
  - 残差参数初值（不同误差模型可能需要不同长度/含义）

示例：

```python
saemix_model_obj = saemix_model(
    model=model1cpt,
    psi0=np.array([[1.5, 30.0, 2.0]]),
    name_modpar=["ka", "V", "CL"],
    transform_par=[1, 1, 1],
    covariance_model=np.eye(3),
    omega_init=np.diag([0.5, 0.5, 0.5]),
    error_model="constant",
)
```

## 3. 控制参数：`saemix_control`

用于设置 SAEM 算法与 MCMC 的迭代策略、随机种子、输出等。

常用入参（见示例脚本）：

- `seed`：随机种子
- `nbiter_saemix=(K1, K2)`：SAEM 两阶段迭代次数
- `display_progress`：是否打印进度
- `warnings`：是否输出警告
- `map`：是否计算 MAP 个体参数估计
- `fim`：是否计算 Fisher 信息矩阵
- `nb_chains`、`nbiter_mcmc=(...)`：MCMC 相关配置（在示例 `demo_estimation.py` 中使用）

示例：

```python
control = saemix_control(
    nbiter_saemix=(50, 30),
    seed=12345,
    display_progress=True,
    warnings=False,
    map=True,
    fim=True,
)
```

## 4. 拟合：`saemix(model, data, control)`

主入口函数，执行 SAEM 拟合并返回结果对象（通常为 `SaemixObject`）。

```python
result = saemix(saemix_model_obj, saemix_data_obj, control)
```

## 5. 结果对象：`SaemixObject`（`result`）

结果对象用于查看估计值、预测、导出与绘图。

你会在示例中看到的常用能力包括：

- `result.summary()`：汇总输出
- `result.predict(type="ppred" | "ipred")`：群体/个体预测
- `result.psi()`：提取个体参数（常用于下游分析）
- `result.results`：原始结果存储（示例中直接读 `mean_phi`、`omega`、`respar` 等）

## 6. 进阶功能

### 6.1 条件分布：`conddist_saemix`

基于 MCMC 估计个体参数的条件分布与 shrinkage 等。

```python
from saemix import conddist_saemix
result = conddist_saemix(result, nsamp=10, max_iter=100, seed=42)
```

### 6.2 模型比较：`compare_saemix`

用于比较不同模型（示例中使用 `method="is"`），输出包含 AIC/BIC 等信息的表。

```python
from saemix import compare_saemix
comparison = compare_saemix(result1, result2, method="is", names=["M1", "M2"])
```

### 6.3 模拟：`simulate_saemix`

从拟合模型模拟数据（可用于 VPC、预测检查等）。

```python
from saemix import simulate_saemix
sim_data = simulate_saemix(result, nsim=100, seed=123, predictions=True, res_var=True)
```

### 6.4 导出：`save_results` / `export_to_csv`

- `save_results(result, directory=..., overwrite=True)`：保存结果到目录
- `export_to_csv(result, path, what=...)`：导出指定内容为 CSV

### 6.5 绘图与 PlotOptions

绘图函数位于对外 API 中（需要安装 `matplotlib`）：

- `plot_gof` / `plot_npde` / `plot_vpc` / `plot_individual_fits` 等

PlotOptions：

- `set_plot_options(...)`
- `get_plot_options()`
- `reset_plot_options()`
