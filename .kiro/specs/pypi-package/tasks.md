# Implementation Plan: PyPI Package Configuration

## Overview

将 saemix Python 项目转换为符合 PyPI 标准的可发布库，使用现代 Python 打包标准。

## Tasks

- [x] 1. 创建版本管理模块
  - [x] 1.1 创建 `saemix/_version.py` 文件
    - 定义 `__version__ = "0.1.0"`
    - 定义 `__version_info__` 元组
    - _Requirements: 2.1, 2.2_
  - [x] 1.2 更新 `saemix/__init__.py` 导入版本
    - 从 `_version` 导入 `__version__` 和 `__version_info__`
    - _Requirements: 2.3_

- [x] 2. 创建 pyproject.toml 配置文件
  - [x] 2.1 配置 build-system 部分
    - 指定 setuptools 和 wheel 作为构建依赖
    - 设置 build-backend
    - _Requirements: 1.1, 1.2_
  - [x] 2.2 配置 project 元数据
    - 设置 name、description、readme、license
    - 配置 dynamic version 从 _version.py 读取
    - 设置 requires-python、authors、keywords
    - _Requirements: 1.3, 5.1, 5.2, 5.5_
  - [x] 2.3 配置依赖声明
    - 声明 dependencies（numpy、pandas、scipy）
    - 配置 optional-dependencies（plot、dev、all）
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  - [x] 2.4 配置 PyPI classifiers
    - 添加开发状态、目标受众、许可证分类器
    - 添加 Python 版本和主题分类器
    - _Requirements: 5.4, 6.3_
  - [x] 2.5 配置 project.urls
    - 添加 Homepage、Documentation、Repository、Issues
    - _Requirements: 5.3_
  - [x] 2.6 配置 setuptools 包发现
    - 设置 packages.find 配置
    - _Requirements: 4.1_
  - [x] 2.7 配置开发工具
    - 添加 pytest.ini_options
    - 添加 black 和 isort 配置
    - _Requirements: 8.1, 8.2_

- [x] 3. 创建许可证和分发文件
  - [x] 3.1 创建 LICENSE 文件
    - 复制 GPL-3 许可证文本
    - _Requirements: 6.1, 6.2_
  - [x] 3.2 创建 MANIFEST.in 文件
    - 包含 LICENSE、README.md、pyproject.toml
    - 包含所有 Python 源文件
    - 排除 __pycache__ 和编译文件
    - _Requirements: 4.3, 6.4_
  - [x] 3.3 创建 CHANGELOG.md 文件
    - 记录初始版本 0.1.0
    - _Requirements: 5.1_

- [x] 4. 更新 setup.py 保持向后兼容
  - [x] 4.1 简化 setup.py
    - 从 _version.py 读取版本
    - 保留基本配置用于向后兼容
    - _Requirements: 1.4, 1.5, 2.4_

- [x] 5. Checkpoint - 验证配置
  - 确保所有配置文件语法正确
  - 验证版本一致性
  - 如有问题请询问用户

- [x] 6. 验证包结构
  - [x] 6.1 验证所有 __init__.py 文件存在
    - 检查 saemix/ 和 saemix/algorithm/ 目录
    - _Requirements: 4.2_
  - [x] 6.2 编写包结构验证测试
    - 测试所有模块可导入
    - 测试 __version__ 属性存在
    - **Property 3: 包结构完整性**
    - **Validates: Requirements 4.1, 4.2, 4.4**

- [x] 7. 测试构建流程
  - [x] 7.1 测试本地安装
    - 运行 `pip install -e .` 验证开发模式安装
    - _Requirements: 8.3_
  - [x] 7.2 编写构建验证测试
    - 测试 sdist 和 wheel 构建
    - 验证分发包内容完整
    - **Property 5: 构建可重复性**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4**

- [x] 8. Final Checkpoint - 完成验证
  - 确保所有测试通过
  - 验证 `twine check dist/*` 通过
  - 如有问题请询问用户

## Notes

- 构建命令：`python -m build`
- 验证命令：`twine check dist/*`
- 上传测试 PyPI：`twine upload --repository testpypi dist/*`
- 上传正式 PyPI：`twine upload dist/*`

