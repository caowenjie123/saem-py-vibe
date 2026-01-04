# 典型工作流

本页用“做事”的角度串联 `saemix` 的常见流程。每一节都对应仓库示例脚本里的实际用法。

## 1. 拟合（Fit）

### 1.1 准备数据

要求是“纵向数据”（long format），每行一条观测，至少包含：

- 个体列（例如 `Id`）
- 预测变量列（例如 `Dose`、`Time`）
- 响应列（例如 `Concentration`）

```python
saemix_data_obj = saemix_data(
    name_data=data,
    name_group="Id",
    name_predictors=["Dose", "Time"],
    name_response="Concentration",
)
```

### 1.2 定义模型函数

要点：

- 模型函数签名固定为 `model(psi, id, xidep)`
- `id` 是内部 0-based 个体索引
- `xidep` 的列顺序与 `name_predictors` 一致

### 1.3 构造模型对象

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

### 1.4 设置控制参数并拟合

```python
control = saemix_control(
    nbiter_saemix=(50, 30),
    seed=12345,
    display_progress=True,
    warnings=False,
    map=True,
    fim=True,
)

result = saemix(saemix_model_obj, saemix_data_obj, control)
```

### 1.5 查看结果与预测

```python
result.summary()

ppred = result.predict(type="ppred")
ipred = result.predict(type="ipred")

psi_est = result.psi()
```

如果你需要更细粒度的结果，示例脚本里也会直接访问：

- `result.results.mean_phi`
- `result.results.omega`
- `result.results.respar`
- `result.results.cond_mean_phi`

## 2. 条件分布（Conditional Distribution）

用于估计个体参数的条件分布、shrinkage 等诊断。

```python
from saemix import conddist_saemix

result = conddist_saemix(result, nsamp=10, max_iter=100, seed=42)

# 示例中会用到：
# result.results.cond_shrinkage
# result.results.cond_mean_phi
```

## 3. 模型比较（Model Comparison）

当你拟合多个模型（例如不同随机效应结构）时，可用信息准则选择更优模型。

```python
from saemix import compare_saemix

comparison = compare_saemix(
    result_full,
    result_reduced,
    method="is",
    names=["Full Model", "Reduced Model"],
)

# comparison 是一个表格（DataFrame-like），示例中会打印 AIC/BIC
```

## 4. 模拟（Simulation）

用拟合得到的模型生成模拟数据，可用于 VPC、预测检查、稳健性分析等。

```python
from saemix import simulate_saemix

sim_data = simulate_saemix(
    result,
    nsim=100,
    seed=123,
    predictions=True,
    res_var=True,
)
```

`sim_data` 通常是一个 DataFrame，包含模拟编号、预测/观测模拟值等列（以实现为准）。

## 5. 结果导出（Export）

将结果落盘，便于复现与下游分析。

```python
from saemix import save_results, export_to_csv

save_results(result, directory="example_results", overwrite=True)
export_to_csv(result, "example_results/individual_params.csv", what="individual")
```

## 6. 绘图（Diagnostics / Plots）

绘图依赖 `matplotlib`（确保已安装）。

常用绘图函数在对外 API 中：

- `plot_observed_vs_pred`
- `plot_residuals`
- `plot_individual_fits`
- `plot_gof`
- `plot_npde`
- `plot_vpc`
- `plot_eta_distributions`

此外你可以通过 `PlotOptions`（或 `set_plot_options`）设置全局绘图参数：

```python
from saemix import set_plot_options, get_plot_options, reset_plot_options

set_plot_options(figsize=(12, 8), dpi=150, alpha=0.7)
options = get_plot_options()
reset_plot_options()
```
