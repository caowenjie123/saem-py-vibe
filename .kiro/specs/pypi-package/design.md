# Design Document: PyPI Package Configuration

## Overview

本设计文档描述如何将 saemix Python 项目转换为符合 PyPI 标准的可发布库。采用现代 Python 打包标准（PEP 517/518/621），使用 `pyproject.toml` 作为主要配置文件，同时保持向后兼容性。

## Architecture

```
saemix/                          # 项目根目录
├── pyproject.toml               # 主配置文件（新增）
├── setup.py                     # 保留用于向后兼容
├── setup.cfg                    # 可选的 setuptools 配置
├── LICENSE                      # 许可证文件（新增）
├── MANIFEST.in                  # 分发包含文件清单（新增）
├── README.md                    # 项目说明
├── CHANGELOG.md                 # 版本变更记录（新增）
├── requirements.txt             # 开发依赖
├── saemix/                      # 主包目录
│   ├── __init__.py              # 包入口（更新版本）
│   ├── _version.py              # 版本单一来源（新增）
│   ├── algorithm/               # 算法子包
│   │   └── __init__.py
│   └── ...其他模块
└── tests/                       # 测试目录
    └── ...
```

## Components and Interfaces

### 1. pyproject.toml 配置结构

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "saemix"
dynamic = ["version"]
description = "SAEM algorithm for nonlinear mixed effects models"
readme = "README.md"
license = {file = "LICENSE"}
requires-python = ">=3.8"
authors = [
    {name = "Author Name", email = "author@example.com"}
]
keywords = ["saem", "mixed-effects", "pharmacometrics", "nlme", "statistics"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Mathematics",
]
dependencies = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
]

[project.optional-dependencies]
plot = ["matplotlib>=3.4.0"]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "hypothesis>=6.0.0",
    "black>=23.0.0",
    "isort>=5.0.0",
    "mypy>=1.0.0",
]
all = ["saemix[plot,dev]"]

[project.urls]
Homepage = "https://github.com/username/saemix-python"
Documentation = "https://github.com/username/saemix-python#readme"
Repository = "https://github.com/username/saemix-python"
Issues = "https://github.com/username/saemix-python/issues"

[tool.setuptools.dynamic]
version = {attr = "saemix._version.__version__"}

[tool.setuptools.packages.find]
where = ["."]
include = ["saemix*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"

[tool.black]
line-length = 88
target-version = ["py38", "py39", "py310", "py311", "py312"]

[tool.isort]
profile = "black"
line_length = 88
```

### 2. 版本管理模块 (_version.py)

```python
# saemix/_version.py
__version__ = "0.1.0"
__version_info__ = tuple(int(x) for x in __version__.split("."))
```

### 3. 更新后的 __init__.py

```python
# saemix/__init__.py
from saemix._version import __version__, __version_info__

# ... 现有导入保持不变
```

### 4. MANIFEST.in 配置

```
include LICENSE
include README.md
include CHANGELOG.md
include pyproject.toml
include setup.py
include requirements.txt
recursive-include saemix *.py
recursive-include tests *.py
global-exclude __pycache__
global-exclude *.pyc
global-exclude *.pyo
```

## Data Models

### 包元数据结构

| 字段 | 类型 | 必需 | 描述 |
|------|------|------|------|
| name | string | 是 | PyPI 包名 |
| version | string | 是 | 语义化版本号 |
| description | string | 是 | 简短描述 |
| readme | string | 是 | README 文件路径 |
| license | object | 是 | 许可证信息 |
| requires-python | string | 是 | Python 版本要求 |
| dependencies | array | 是 | 必需依赖列表 |
| optional-dependencies | object | 否 | 可选依赖组 |
| classifiers | array | 是 | PyPI 分类器 |
| urls | object | 否 | 项目链接 |

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: 版本一致性

*For any* valid package build, the version exposed via `saemix.__version__` SHALL equal the version in `pyproject.toml` and the built distribution metadata.

**Validates: Requirements 2.1, 2.4**

### Property 2: 依赖完整性

*For any* fresh Python environment with only the declared dependencies installed, importing `saemix` and calling its core functions SHALL succeed without ImportError.

**Validates: Requirements 3.1, 3.3**

### Property 3: 包结构完整性

*For any* built distribution (sdist or wheel), all Python modules under `saemix/` directory SHALL be included and importable.

**Validates: Requirements 4.1, 4.2, 4.4**

### Property 4: 元数据完整性

*For any* built distribution, the package metadata SHALL contain all required fields (name, version, description, author, license) with non-empty values.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4**

### Property 5: 构建可重复性

*For any* given source tree state, building sdist then building wheel from that sdist SHALL produce functionally equivalent packages.

**Validates: Requirements 7.1, 7.2, 7.4**

## Error Handling

### 构建错误

| 错误类型 | 原因 | 处理方式 |
|----------|------|----------|
| MissingDependency | 缺少构建依赖 | 安装 build-system.requires |
| InvalidVersion | 版本格式错误 | 检查 _version.py 格式 |
| MissingFile | MANIFEST.in 引用缺失文件 | 更新 MANIFEST.in |
| ImportError | 模块导入失败 | 检查 __init__.py 和依赖 |

### 安装错误

| 错误类型 | 原因 | 处理方式 |
|----------|------|----------|
| DependencyConflict | 依赖版本冲突 | 调整版本约束 |
| PythonVersionError | Python 版本不兼容 | 检查 requires-python |

## Testing Strategy

### 单元测试

- 验证 `__version__` 属性存在且格式正确
- 验证所有公开 API 可导入
- 验证可选依赖不影响核心功能

### 属性测试

- **Property 1**: 构建后检查版本一致性
- **Property 2**: 在干净环境中测试导入
- **Property 3**: 检查分发包内容
- **Property 4**: 验证元数据字段
- **Property 5**: 比较 sdist 和 wheel 构建结果

### 集成测试

- 使用 `pip install .` 测试本地安装
- 使用 `pip install -e .` 测试开发模式安装
- 使用 `python -m build` 测试构建流程
- 使用 `twine check` 验证分发包

### 测试命令

```bash
# 运行测试
pytest tests/

# 构建分发包
python -m build

# 验证分发包
twine check dist/*

# 测试安装
pip install dist/saemix-*.whl
python -c "import saemix; print(saemix.__version__)"
```

