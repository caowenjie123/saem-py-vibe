# 安装与环境

## Python 版本

本项目在 `pyproject.toml` 中声明 `Python >= 3.8`。

## 方式一：从源码安装（推荐用于本仓库）

在项目根目录执行：

```bash
pip install -r requirements.txt
pip install -e .
```

- `pip install -e .` 为开发模式安装，便于你在本仓库内调试与运行示例。

## 方式二：直接安装依赖并运行脚本

如果你只是想运行仓库中的示例脚本，也可以只安装依赖：

```bash
pip install -r requirements.txt
```

然后运行：

```bash
python demo_estimation.py
python examples/basic_example.py
```

## 可选依赖：绘图（matplotlib）

绘图相关功能需要 `matplotlib`。

- 通过 pip 安装：

```bash
pip install matplotlib
```

- 或使用项目声明的可选依赖（当你以包方式安装时）：

```bash
pip install ".[plot]"
```

## 开发依赖（测试/格式化）

如需运行测试与开发工具：

```bash
pip install ".[dev]"
pytest
```

## 常见安装问题

- **在 Windows 上建议使用虚拟环境**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

- **依赖冲突**：建议新建干净虚拟环境，或升级 pip：

```bash
python -m pip install --upgrade pip
```
