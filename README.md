# saemix-python

Python 实现的 SAEM（Stochastic Approximation Expectation Maximization）算法库，用于非线性混合效应模型的参数估计。本项目是 R 语言 [saemix](https://github.com/saemixdevelopment/saemix) 包的 Python 移植版本。

## 功能特性

- 完整的 SAEM 算法实现，用于非线性混合效应模型参数估计
- 支持多种误差模型：constant、proportional、combined、exponential
- 支持参数变换：normal、log-normal、probit、logit
- 支持协变量建模
- MAP（最大后验）个体参数估计
- Fisher 信息矩阵计算
- 多种似然估计方法（重要性采样、高斯积分）
- 丰富的诊断图功能：残差图、VPC、NPDE 等

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e .
```

### 依赖项

- Python >= 3.7
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib（可选，用于绑图功能）

## 快速开始

```python
import numpy as np
import pandas as pd
from saemix import saemix, saemix_data, saemix_model, saemix_control

# 1. 定义模型函数（注意：Python 使用 0-based 索引）
def model1cpt(psi, id, xidep):
    """一室模型，一级吸收"""
    dose = xidep[:, 0]    # 第 0 列：剂量
    tim = xidep[:, 1]     # 第 1 列：时间
    ka = psi[id, 0]       # 第 0 个参数：吸收速率常数
    V = psi[id, 1]        # 第 1 个参数：分布容积
    CL = psi[id, 2]       # 第 2 个参数：清除率
    k = CL / V
    ypred = dose * ka / (V * (ka - k)) * (np.exp(-k * tim) - np.exp(-ka * tim))
    return ypred

# 2. 准备数据
data = pd.DataFrame({
    'Id': [1, 1, 1, 2, 2, 2],
    'Dose': [100, 0, 0, 100, 0, 0],
    'Time': [0, 1, 2, 0, 1, 2],
    'Concentration': [0, 5, 3, 0, 6, 4]
})

# 3. 创建数据对象
saemix_data_obj = saemix_data(
    name_data=data,
    name_group='Id',
    name_predictors=['Dose', 'Time'],
    name_response='Concentration'
)

# 4. 创建模型对象
saemix_model_obj = saemix_model(
    model=model1cpt,
    description="One-compartment model",
    psi0=np.array([[1.0, 20.0, 0.5]]),
    name_modpar=["ka", "V", "CL"],
    transform_par=[1, 1, 1],  # log-normal 变换
    covariance_model=np.eye(3),
    omega_init=np.eye(3),
    error_model="constant"
)

# 5. 设置控制参数
control = saemix_control(
    seed=632545,
    nbiter_saemix=(300, 100),
    display_progress=True
)

# 6. 运行 SAEM 算法
result = saemix(saemix_model_obj, saemix_data_obj, control)

# 7. 查看结果
print(result)
result.summary()

# 8. 提取个体参数
psi_est = result.psi()
```

## 核心 API

### SaemixData

用于管理和验证纵向数据：

```python
saemix_data(
    name_data,           # DataFrame 或文件路径
    name_group,          # 个体 ID 列名
    name_predictors,     # 预测变量列名列表
    name_response,       # 响应变量列名
    name_covariates=None,# 协变量列名列表（可选）
    name_X=None,         # 绘图用 X 轴变量名（可选）
    units=None,          # 单位字典（可选）
)
```

### SaemixModel

用于定义非线性混合效应模型：

```python
saemix_model(
    model,               # 模型函数 f(psi, id, xidep)
    psi0,                # 初始参数矩阵
    description="",      # 模型描述
    modeltype="structural",  # 模型类型
    error_model="constant",  # 误差模型
    transform_par=None,  # 参数变换类型
    fixed_estim=None,    # 是否估计参数
    covariate_model=None,# 协变量模型矩阵
    covariance_model=None,# 协方差模型矩阵
    omega_init=None,     # Omega 初始值
    name_modpar=None,    # 参数名称列表
)
```

### saemix_control

算法控制参数：

```python
saemix_control(
    map=True,            # 是否计算 MAP 估计
    fim=True,            # 是否计算 Fisher 信息矩阵
    ll_is=False,         # 是否计算重要性采样似然
    nbiter_saemix=(300, 100),  # SAEM 迭代次数 [K1, K2]
    seed=23456,          # 随机种子
    display_progress=False,    # 是否显示进度
)
```

## 诊断图

```python
from saemix import (
    plot_observed_vs_pred,
    plot_residuals,
    plot_individual_fits,
    plot_gof,
    plot_npde,
    plot_vpc,
    plot_eta_distributions,
)

# 观测值 vs 预测值
plot_observed_vs_pred(result)

# 残差图
plot_residuals(result)

# 个体拟合图
plot_individual_fits(result, n=6)

# 综合 GOF 图（4 面板）
plot_gof(result)

# NPDE 诊断图
plot_npde(result, nsim=1000)

# VPC 图
plot_vpc(result, nsim=1000, bins=10)

# 随机效应分布图
plot_eta_distributions(result)
```

## 与 R saemix 的主要差异

| 特性 | R saemix | Python saemix |
|------|----------|---------------|
| 索引 | 1-based | 0-based |
| 命名约定 | camelCase | snake_case |
| 数据结构 | data.frame | pandas.DataFrame |
| 矩阵 | matrix | numpy.ndarray |

### 模型函数索引转换示例

```r
# R 版本（1-based 索引）
model1cpt <- function(psi, id, xidep) {
    dose <- xidep[, 1]    # 第 1 列
    tim <- xidep[, 2]     # 第 2 列
    ka <- psi[id, 1]      # 第 1 个参数
    V <- psi[id, 2]       # 第 2 个参数
    CL <- psi[id, 3]      # 第 3 个参数
    # ...
}
```

```python
# Python 版本（0-based 索引）
def model1cpt(psi, id, xidep):
    dose = xidep[:, 0]    # 第 0 列
    tim = xidep[:, 1]     # 第 1 列
    ka = psi[id, 0]       # 第 0 个参数
    V = psi[id, 1]        # 第 1 个参数
    CL = psi[id, 2]       # 第 2 个参数
    # ...
```

## 项目结构

```
saemix/
├── __init__.py          # 包入口
├── data.py              # SaemixData 类
├── model.py             # SaemixModel 类
├── control.py           # 控制参数
├── results.py           # 结果对象
├── main.py              # 主函数 saemix()
├── diagnostics.py       # 诊断图功能
├── utils.py             # 工具函数
└── algorithm/           # 算法实现
    ├── saem.py          # SAEM 主算法
    ├── estep.py         # E 步
    ├── mstep.py         # M 步
    ├── initialization.py# 初始化
    ├── map_estimation.py# MAP 估计
    ├── fim.py           # Fisher 信息矩阵
    ├── likelihood.py    # 似然计算
    └── predict.py       # 预测
```

## 参考文献

- Kuhn, E., & Lavielle, M. (2005). Maximum likelihood estimation in nonlinear mixed effects models. Computational Statistics & Data Analysis, 49(4), 1020-1038.
- Comets, E., Lavenu, A., & Lavielle, M. (2017). Parameter estimation in nonlinear mixed effect models using saemix, an R implementation of the SAEM algorithm. Journal of Statistical Software, 80(3), 1-41.

## 许可证

本项目遵循与 R saemix 包相同的许可证。
