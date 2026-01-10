# Requirements Document

## Introduction

将现有的 saemix Python 项目转换为符合 PyPI 标准的可发布 Python 库。该库实现了 SAEM（Stochastic Approximation Expectation Maximization）算法，用于非线性混合效应模型的参数估计。

## Glossary

- **PyPI**: Python Package Index，Python 官方的第三方包仓库
- **Package_Builder**: 负责构建和打包 Python 库的系统组件
- **Metadata_Manager**: 管理包元数据（版本、作者、许可证等）的组件
- **Distribution_System**: 负责生成 sdist 和 wheel 分发包的系统
- **Validator**: 验证包配置和结构正确性的组件

## Requirements

### Requirement 1: 现代化包配置

**User Story:** As a developer, I want to use modern Python packaging standards (pyproject.toml), so that the package follows current best practices and is compatible with modern build tools.

#### Acceptance Criteria

1. THE Package_Builder SHALL use pyproject.toml as the primary configuration file
2. THE Package_Builder SHALL specify build-system requirements using PEP 517/518 standards
3. THE Metadata_Manager SHALL include all required PyPI metadata fields (name, version, description, author, license, classifiers)
4. THE Package_Builder SHALL support both setuptools and modern build backends
5. WHEN setup.py exists, THE Package_Builder SHALL maintain backward compatibility

### Requirement 2: 版本管理

**User Story:** As a maintainer, I want proper version management, so that users can track releases and dependencies correctly.

#### Acceptance Criteria

1. THE Metadata_Manager SHALL define version in a single source of truth location
2. THE Metadata_Manager SHALL follow semantic versioning (MAJOR.MINOR.PATCH)
3. WHEN the package is imported, THE Package_Builder SHALL expose __version__ attribute
4. THE Validator SHALL ensure version consistency across all configuration files

### Requirement 3: 依赖声明

**User Story:** As a user, I want clear dependency declarations, so that I can install the package with all required dependencies automatically.

#### Acceptance Criteria

1. THE Package_Builder SHALL declare all required dependencies with version constraints
2. THE Package_Builder SHALL separate optional dependencies into extras (e.g., [dev], [plot])
3. THE Package_Builder SHALL specify Python version requirements
4. WHEN optional features are needed, THE Package_Builder SHALL allow installation via extras_require

### Requirement 4: 包结构验证

**User Story:** As a developer, I want the package structure to be validated, so that all modules are correctly included in the distribution.

#### Acceptance Criteria

1. THE Validator SHALL ensure all Python modules are discoverable
2. THE Validator SHALL verify __init__.py files exist in all package directories
3. THE Package_Builder SHALL include all necessary package data files
4. WHEN building distribution, THE Validator SHALL check for missing files

### Requirement 5: 文档和元数据

**User Story:** As a potential user, I want comprehensive package metadata, so that I can understand what the package does before installing.

#### Acceptance Criteria

1. THE Metadata_Manager SHALL include a long description from README.md
2. THE Metadata_Manager SHALL specify content type as text/markdown
3. THE Metadata_Manager SHALL include project URLs (homepage, repository, documentation)
4. THE Metadata_Manager SHALL include appropriate PyPI classifiers for discoverability
5. THE Metadata_Manager SHALL specify keywords for search optimization

### Requirement 6: 许可证合规

**User Story:** As a user, I want clear licensing information, so that I know how I can use the package.

#### Acceptance Criteria

1. THE Metadata_Manager SHALL include a LICENSE file in the distribution
2. THE Metadata_Manager SHALL specify license in pyproject.toml
3. THE Package_Builder SHALL include license classifier in PyPI metadata
4. WHEN the package is distributed, THE Package_Builder SHALL include license text

### Requirement 7: 构建和分发

**User Story:** As a maintainer, I want to build source and wheel distributions, so that users can install the package efficiently.

#### Acceptance Criteria

1. THE Distribution_System SHALL generate source distribution (sdist)
2. THE Distribution_System SHALL generate wheel distribution (bdist_wheel)
3. THE Distribution_System SHALL create platform-independent wheels (pure Python)
4. WHEN building, THE Validator SHALL verify distribution contents are complete

### Requirement 8: 质量保证配置

**User Story:** As a developer, I want development tool configurations, so that code quality can be maintained.

#### Acceptance Criteria

1. THE Package_Builder SHALL include pytest configuration for testing
2. THE Package_Builder SHALL include configuration for code formatting tools
3. THE Package_Builder SHALL support editable installation for development
4. WHEN running tests, THE Validator SHALL use the configured test framework

