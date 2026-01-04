# 快速开始

本页给出一个最小可运行的 `saemix` 工作流：

- 准备纵向数据（`pandas.DataFrame`）
- 定义模型函数 `model(psi, id, xidep)`
- 构造 `saemix_data`、`saemix_model`、`saemix_control`
- 调用 `saemix()` 拟合

## 1. 运行仓库示例

在项目根目录：

```bash
python demo_estimation.py
python examples/basic_example.py
```

- `demo_estimation.py`：线性模型的参数估计演示
- `examples/basic_example.py`：Theophylline PK 分析，包含拟合、条件分布、模型比较、模拟、导出等流程

## 2. 最小代码示例

将以下内容保存为 `my_fit.py` 并运行 `python my_fit.py`：

```python
import numpy as np
import pandas as pd

from saemix import saemix, saemix_data, saemix_model, saemix_control


def model1cpt(psi, id, xidep):
    # xidep 的列顺序由 saemix_data(name_predictors=...) 决定
    dose = xidep[:, 0]
    tim = xidep[:, 1]

    # 注意：Python 为 0-based 索引
    ka = psi[id, 0]
    V = psi[id, 1]
    CL = psi[id, 2]

    k = CL / V
    ypred = dose * ka / (V * (ka - k)) * (np.exp(-k * tim) - np.exp(-ka * tim))
    return ypred


data = pd.DataFrame(
    {
        "Id": [1, 1, 1, 2, 2, 2],
        "Dose": [100, 0, 0, 100, 0, 0],
        "Time": [0, 1, 2, 0, 1, 2],
        "Concentration": [0, 5, 3, 0, 6, 4],
    }
)

saemix_data_obj = saemix_data(
    name_data=data,
    name_group="Id",
    name_predictors=["Dose", "Time"],
    name_response="Concentration",
)

saemix_model_obj = saemix_model(
    model=model1cpt,
    description="One-compartment model",
    psi0=np.array([[1.0, 20.0, 0.5]]),
    name_modpar=["ka", "V", "CL"],
    transform_par=[1, 1, 1],
    covariance_model=np.eye(3),
    omega_init=np.eye(3),
    error_model="constant",
)

control = saemix_control(
    seed=632545,
    nbiter_saemix=(300, 100),
    display_progress=True,
)

result = saemix(saemix_model_obj, saemix_data_obj, control)

print(result)
result.summary()

psi_est = result.psi()
print("psi (individual params) shape:", psi_est.shape)
```

## 3. 注意事项

- `model(psi, id, xidep)` 中的 `id` 是当前个体的 **0-based** 索引（`0..N-1`），不是原始数据中的 `Id` 值。
- `xidep` 的列顺序与 `saemix_data(..., name_predictors=[...])` 保持一致。
- 如果你使用了对数正态等变换（`transform_par=[1, ...]`），初始值 `psi0` 通常填“原始尺度”的正值即可（建议结合具体模型理解与调参）。
