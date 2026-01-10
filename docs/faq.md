# FAQ / 常见问题排查

## 1. `ModuleNotFoundError: No module named 'saemix'`

- 确认你在项目根目录执行过：

```bash
pip install -e .
```

- 或者确认你运行脚本的 Python 解释器，就是你安装依赖的那个环境。

## 2. 绘图时报错：`ImportError: No module named 'matplotlib'`

绘图为可选依赖，请安装：

```bash
pip install matplotlib
```

或：

```bash
pip install ".[plot]"
```

## 3. 模型函数里 `id` 为什么不是数据里的 `Id`？

`model(psi, id, xidep)` 里的 `id` 是内部重映射后的 **0-based** 个体索引（`0..N-1`）。

- 你的原始数据里 `Id` 可以是 1..N、也可以是不连续的编码
- `saemix_data` 通常会将其映射为内部索引，供算法高效计算

因此在模型函数中写 `psi[id, 0]` 是正确的。

## 4. 模型计算出现 `divide by zero` / `nan` / 数值爆炸

常见原因：

- 初值 `psi0` 不合理（例如应该为正的参数被给成 0/负数）
- 模型表达式存在除以接近 0 的项（例如一室模型里 `ka - k` 很小）

建议：

- 参考 `examples/basic_example.py` 中对 `ka` 与 `k` 做数值安全处理的写法
- 通过 `transform_par` 对参数做合适的变换（例如 log-normal）
- 调整 `psi0` 与 `omega_init` 到更合理的量级

## 5. 结果不收敛/估计值不稳定

可尝试：

- 增加 `nbiter_saemix=(K1, K2)`
- 固定部分参数或减少随机效应（调整 `covariance_model`）
- 改善初值 `psi0`、`omega_init`、残差初值（如 `error_init`）
- 增加 `nb_chains` 或调整 `nbiter_mcmc`（如果你在使用 MCMC 相关配置）

## 6. 运行很慢

SAEM + MCMC 在复杂模型/大样本下可能计算量较大。

建议：

- 先用较小的 `nbiter_saemix` 做可行性验证
- 在“模型比较”阶段用较少迭代快速筛选结构，确定后再加大迭代做最终拟合
- 避免在模型函数中写过慢的 Python 循环，尽量使用 `numpy` 向量化

## 7. 如何复现实验？

- 在 `saemix_control` 中设置 `seed=...`
- 确保你的依赖版本一致（可在虚拟环境里固定版本）

## 8. 我该从哪里看更完整的接口？

- 入口导出列表可看 `saemix/__init__.py`（对外暴露了哪些函数/类）
- 复杂用法建议直接阅读示例：
  - `demo_estimation.py`
  - `examples/basic_example.py`
- 项目的总体设计与 R/Python 差异说明可参考 `DESIGN.md`
