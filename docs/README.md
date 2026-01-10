# saemix 使用文档

本目录提供 `saemix`（SAEM：Stochastic Approximation EM）Python 实现的使用说明。

## 文档目录

- [安装与环境](installation.md)
- [快速开始](quickstart.md)
- [核心概念与 API 总览](api_overview.md)
- [典型工作流（拟合/条件分布/比较/模拟/导出/绘图）](workflow.md)
- [FAQ / 常见问题排查](faq.md)

## 适用范围

- 本项目的最低 Python 版本要求见 `pyproject.toml`（当前为 `>=3.8`）。
- 依赖主要为 `numpy`、`pandas`、`scipy`，绘图功能需要可选依赖 `matplotlib`。

## 你可能最常用的两个入口

- 运行示例脚本

```bash
python demo_estimation.py
python examples/basic_example.py
```

- 在你自己的脚本中调用

```python
from saemix import saemix, saemix_data, saemix_model, saemix_control
```
