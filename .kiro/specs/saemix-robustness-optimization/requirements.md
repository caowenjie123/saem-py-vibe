# Requirements Document

## Introduction

本文档定义了 Python 版本 saemix 库的鲁棒性与工程化优化需求。本次优化旨在提升代码的正确性、可复现性、数值稳定性和工程化质量，同时不破坏现有 API。所有可能影响拟合结果或随机性的改动都必须配套回归测试。

## Glossary

- **SaemixData**: 数据管理类，负责加载、验证和处理输入数据
- **SaemixControl**: 控制参数类，管理 SAEM 算法的运行参数
- **RNG**: 随机数生成器 (Random Number Generator)，用于可复现的随机数生成
- **MDV**: Missing Dependent Variable，缺失因变量标记列
- **CENS**: Censoring，删失数据标记列
- **OCC**: Occasion，场合标记列
- **YTYPE**: Y Type，响应类型标记列
- **Covariate**: 协变量，影响个体参数的外部变量
- **Parameter_Transform**: 参数变换，如 log、logit 等用于约束参数范围的变换
- **CI**: Continuous Integration，持续集成

## Requirements

### Requirement 1: SaemixData 辅助列处理修复

**User Story:** As a pharmacometrician, I want the data class to correctly handle auxiliary columns (mdv, cens, occ, ytype), so that my data specifications are properly applied during model fitting.

#### Acceptance Criteria

1. WHEN a user provides column names for mdv, cens, occ, or ytype in SaemixData constructor, THE SaemixData SHALL correctly map these columns after data subsetting and filtering
2. WHEN the specified auxiliary column does not exist in the data, THE SaemixData SHALL raise a clear ValueError with the missing column name
3. WHEN verbose mode is enabled, THE SaemixData SHALL print which columns are being used for mdv, cens, occ, and ytype
4. WHEN verbose mode is enabled and default values are used, THE SaemixData SHALL print a message indicating default values are being applied
5. WHEN auxiliary columns are specified, THE SaemixData SHALL preserve the column data through all internal data transformations

### Requirement 2: 协变量校验逻辑修复

**User Story:** As a user, I want the covariate validation to work correctly, so that missing or invalid covariates are handled predictably without silent failures.

#### Acceptance Criteria

1. WHEN validating covariates, THE SaemixData SHALL use a new list construction approach instead of modifying the list during iteration
2. WHEN a specified covariate column does not exist in the data, THE SaemixData SHALL remove it from the covariate list and issue a warning
3. WHEN all specified covariates are missing, THE SaemixData SHALL set the covariate list to empty and issue a warning
4. WHEN verbose mode is enabled, THE SaemixData SHALL print which covariates were found and which were ignored
5. WHEN covariate validation completes, THE SaemixData SHALL have a stable and predictable covariate list

### Requirement 3: 统一随机数管理

**User Story:** As a researcher, I want reproducible results with the same seed, so that I can verify and share my analyses reliably.

#### Acceptance Criteria

1. WHEN a user provides a seed parameter to saemix or SaemixControl, THE SAEM_Algorithm SHALL create a numpy.random.Generator instance for all random operations
2. WHEN the RNG is created, THE SAEM_Algorithm SHALL pass it to all modules that require random numbers (simulation, conddist, etc.)
3. WHEN no seed is provided, THE SAEM_Algorithm SHALL create a default RNG without affecting the global numpy random state
4. THE SAEM_Algorithm SHALL NOT call np.random.seed() or modify the global random state
5. WHEN the same seed is used with identical inputs, THE SAEM_Algorithm SHALL produce identical results within numerical precision
6. WHEN a user's code sets np.random.seed() before calling saemix, THE SAEM_Algorithm SHALL NOT affect that global state after execution

### Requirement 4: 参数变换数值防护

**User Story:** As a modeler, I want parameter transformations to handle edge cases gracefully, so that I get informative errors instead of silent NaN/Inf values.

#### Acceptance Criteria

1. WHEN applying log transformation, THE Parameter_Transform SHALL clip input values to a minimum epsilon (e.g., 1e-10) to prevent -Inf
2. WHEN applying logit transformation, THE Parameter_Transform SHALL clip input values to (epsilon, 1-epsilon) to prevent -Inf and +Inf
3. WHEN a transformation produces NaN or Inf despite clipping, THE Parameter_Transform SHALL raise a ValueError with diagnostic information
4. WHEN verbose mode is enabled and clipping is applied, THE Parameter_Transform SHALL issue a warning indicating which values were clipped
5. WHEN inverse transformations are applied, THE Parameter_Transform SHALL apply corresponding bounds to prevent invalid outputs

### Requirement 5: 关键计算路径数值稳定性

**User Story:** As a user, I want clear error messages when numerical issues occur, so that I can diagnose and fix problems in my model or data.

#### Acceptance Criteria

1. WHEN a matrix operation fails (e.g., singular matrix), THE Algorithm SHALL raise a descriptive error indicating the operation and possible causes
2. WHEN computing log-likelihood produces NaN or Inf, THE Algorithm SHALL raise a ValueError with the iteration number and parameter values
3. WHEN the Cholesky decomposition fails, THE Algorithm SHALL provide guidance on checking the covariance matrix specification
4. WHEN eigenvalue computation produces negative values for a covariance matrix, THE Algorithm SHALL issue a warning and attempt correction
5. WHEN numerical issues occur, THE Algorithm SHALL include the input ranges in the error message for debugging

### Requirement 6: CI 工作流建立

**User Story:** As a developer, I want automated testing and build verification, so that code quality is maintained across contributions.

#### Acceptance Criteria

1. WHEN code is pushed to the main branch, THE CI_System SHALL run pytest on all supported Python versions
2. WHEN a pull request is created, THE CI_System SHALL automatically trigger the test suite
3. WHEN tests are run, THE CI_System SHALL verify that python -m build succeeds
4. WHEN any test fails, THE CI_System SHALL report the failure clearly in the PR status
5. WHEN the CI pipeline completes successfully, THE CI_System SHALL show a green status indicator

### Requirement 7: API 一致性与错误暴露

**User Story:** As a user, I want consistent API behavior and clear error messages, so that I can use the library reliably and debug issues easily.

#### Acceptance Criteria

1. WHEN an optional dependency is missing, THE Package SHALL disable only the related functionality and provide a clear error message when that functionality is accessed
2. WHEN the package is imported, THE Package SHALL NOT silently swallow ImportError for critical dependencies
3. WHEN ytype normalization is performed, THE Package SHALL use a single unified implementation across all modules
4. WHEN a user accesses internal 0-based indices, THE Package SHALL provide helper functions to convert between user-facing 1-based IDs and internal indices
5. WHEN an error occurs, THE Package SHALL provide actionable error messages that suggest possible solutions

### Requirement 8: 回归测试体系

**User Story:** As a maintainer, I want regression tests that catch unintended changes, so that I can confidently make improvements without breaking existing functionality.

#### Acceptance Criteria

1. WHEN a regression test suite is established, THE Test_Suite SHALL include at least one classic dataset (e.g., theo.saemix.tab) with fixed parameters and seed
2. WHEN regression tests run, THE Test_Suite SHALL verify that key outputs (fixed effects, omega, residual parameters, LL, AIC, BIC) are within acceptable tolerance
3. WHEN a code change causes regression test failure, THE Test_Suite SHALL clearly indicate which metrics changed and by how much
4. WHEN new features are added, THE Test_Suite SHALL require corresponding regression test updates if outputs change
5. WHEN tolerance bounds are defined, THE Test_Suite SHALL document the rationale for each tolerance value

