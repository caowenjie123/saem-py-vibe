# Requirements Document

## Introduction

本文档定义了 Python 版本 saemix 库的功能增强需求。saemix 是一个用于非线性混合效应模型参数估计的统计软件包，基于 SAEM（随机近似期望最大化）算法。本次增强旨在将 Python 版本的功能与 R 版本对齐，实现条件分布估计、模型比较、逐步回归等核心功能。

## Glossary

- **SAEM_Algorithm**: 随机近似期望最大化算法，用于非线性混合效应模型的参数估计
- **SaemixObject**: 包含拟合结果的主要对象，存储模型、数据和估计结果
- **SaemixRes**: 结果类，存储参数估计值、标准误、置信区间等
- **Conditional_Distribution**: 给定观测数据条件下个体参数的后验分布
- **MCMC_Sampler**: 马尔可夫链蒙特卡洛采样器，用于从条件分布中采样
- **Information_Criterion**: 模型选择准则，包括 AIC 和 BIC
- **Stepwise_Regression**: 逐步回归方法，用于协变量和随机效应的自动选择
- **Fisher_Information_Matrix**: Fisher 信息矩阵，用于计算参数估计的标准误
- **Shrinkage**: 收缩估计，衡量个体参数向总体均值收缩的程度
- **VPC**: 视觉预测检验图，用于模型诊断
- **NPDE**: 归一化预测分布误差，用于模型评估

## Requirements

### Requirement 1: 条件分布估计

**User Story:** As a pharmacometrician, I want to estimate the conditional distribution of individual parameters, so that I can obtain more accurate individual parameter estimates and assess parameter uncertainty.

#### Acceptance Criteria

1. WHEN a user calls the conditional distribution function with a fitted SaemixObject, THE Conditional_Distribution_Estimator SHALL compute the conditional mean of individual parameters using MCMC sampling
2. WHEN the MCMC sampling is performed, THE Conditional_Distribution_Estimator SHALL compute the conditional variance of individual parameters
3. WHEN the conditional distribution is estimated, THE Conditional_Distribution_Estimator SHALL calculate the conditional shrinkage for each parameter
4. WHEN the user specifies nsamp parameter, THE Conditional_Distribution_Estimator SHALL generate the specified number of samples from the conditional distribution
5. WHEN the user sets max_iter parameter, THE Conditional_Distribution_Estimator SHALL limit the MCMC iterations to the specified value
6. WHEN the user enables plot option, THE Conditional_Distribution_Estimator SHALL display convergence diagnostic plots
7. WHEN the estimation completes, THE Conditional_Distribution_Estimator SHALL return an updated SaemixObject containing cond_mean_phi, cond_var_phi, cond_shrinkage, and phi_samp attributes

### Requirement 2: 模型比较

**User Story:** As a modeler, I want to compare multiple fitted models using information criteria, so that I can select the best model for my data.

#### Acceptance Criteria

1. WHEN a user provides multiple SaemixObject instances to the compare function, THE Model_Comparator SHALL compute AIC for each model
2. WHEN comparing models, THE Model_Comparator SHALL compute BIC for each model
3. WHEN comparing models, THE Model_Comparator SHALL compute BIC.cov (covariate-specific BIC) for each model
4. WHEN the user specifies a likelihood method (is, lin, gq), THE Model_Comparator SHALL use that method for likelihood calculation
5. WHEN the comparison completes, THE Model_Comparator SHALL return a DataFrame containing model names, number of parameters, log-likelihood, AIC, BIC, and BIC.cov values
6. WHEN models have different data, THE Model_Comparator SHALL raise a ValueError with a descriptive message

### Requirement 3: 结果对象完善

**User Story:** As a user, I want comprehensive result objects, so that I can access all estimation outputs including confidence intervals and iteration history.

#### Acceptance Criteria

1. WHEN parameter estimation completes, THE SaemixRes SHALL compute and store 95% confidence intervals for all fixed effect parameters
2. WHEN the SAEM algorithm runs, THE SaemixRes SHALL record population parameters at each iteration in parpop attribute
3. WHEN the SAEM algorithm runs, THE SaemixRes SHALL record all parameters (including random effects) at each iteration in allpar attribute
4. WHEN predictions are computed, THE SaemixRes SHALL store predictions in a DataFrame with columns for id, time, observed, predicted, and residuals
5. WHEN residuals are computed, THE SaemixRes SHALL calculate and store individual residuals (ires), weighted residuals (wres), and prediction discrepancies (pd)
6. WHEN the user accesses conf_int property, THE SaemixRes SHALL return a DataFrame with parameter names, estimates, standard errors, and confidence bounds

### Requirement 4: 逐步回归

**User Story:** As a pharmacometrician, I want to perform stepwise covariate selection, so that I can automatically identify significant covariates and random effects.

#### Acceptance Criteria

1. WHEN a user calls forward_procedure, THE Stepwise_Selector SHALL iteratively add covariates that improve BIC until no improvement is found
2. WHEN a user calls backward_procedure, THE Stepwise_Selector SHALL iteratively remove covariates that do not significantly worsen BIC until no more can be removed
3. WHEN a user calls stepwise_procedure with direction='both', THE Stepwise_Selector SHALL alternate between forward and backward steps
4. WHEN trace parameter is True, THE Stepwise_Selector SHALL print the selection process including BIC values at each step
5. WHEN the selection completes, THE Stepwise_Selector SHALL return a SaemixObject fitted with the optimal covariate model
6. WHEN the user provides covariate_init parameter, THE Stepwise_Selector SHALL start from the specified initial covariate model
7. WHEN no covariates improve the model, THE Stepwise_Selector SHALL return the original model unchanged

### Requirement 5: 模拟功能完善

**User Story:** As a researcher, I want to simulate data from fitted models, so that I can perform simulation studies and model validation.

#### Acceptance Criteria

1. WHEN a user calls simulate_saemix with a fitted SaemixObject, THE Simulator SHALL generate simulated observations using the estimated parameters
2. WHEN nsim parameter is specified, THE Simulator SHALL generate the specified number of simulation replicates
3. WHEN seed parameter is provided, THE Simulator SHALL use it for reproducible random number generation
4. WHEN predictions parameter is True, THE Simulator SHALL include population and individual predictions in the output
5. WHEN res_var parameter is True, THE Simulator SHALL add residual variability to the simulated observations
6. WHEN simulating discrete responses, THE Simulator SHALL use the user-provided simulate_function for generating discrete outcomes
7. WHEN simulation completes, THE Simulator SHALL return a DataFrame containing simulated data with replicate identifiers

### Requirement 6: 诊断图增强

**User Story:** As a modeler, I want comprehensive diagnostic plots, so that I can thoroughly evaluate model fit and convergence.

#### Acceptance Criteria

1. WHEN a user calls plot_convergence, THE Diagnostics_Plotter SHALL display parameter estimates versus iteration number for all estimated parameters
2. WHEN a user calls plot_likelihood, THE Diagnostics_Plotter SHALL display the log-likelihood trajectory during estimation
3. WHEN a user calls plot_parameters_vs_covariates, THE Diagnostics_Plotter SHALL display scatter plots or box plots of parameters against covariates
4. WHEN a user calls plot_randeff_vs_covariates, THE Diagnostics_Plotter SHALL display random effects versus covariate values
5. WHEN a user calls plot_marginal_distribution, THE Diagnostics_Plotter SHALL display histograms or density plots of parameter distributions
6. WHEN a user calls plot_correlations, THE Diagnostics_Plotter SHALL display a correlation matrix plot of random effects
7. WHEN the user specifies figure size or style options, THE Diagnostics_Plotter SHALL apply the specified formatting

### Requirement 7: 结果保存与导出

**User Story:** As a user, I want to save and export results, so that I can archive my analyses and share results with collaborators.

#### Acceptance Criteria

1. WHEN a user calls save_results with a SaemixObject, THE Result_Exporter SHALL save all estimation results to the specified directory
2. WHEN saving results, THE Result_Exporter SHALL create separate files for parameters, predictions, and diagnostics
3. WHEN a user calls export_to_csv, THE Result_Exporter SHALL export the specified result component to a CSV file
4. WHEN saving plots, THE Result_Exporter SHALL save diagnostic plots in the specified format (png, pdf, svg)
5. WHEN the output directory does not exist, THE Result_Exporter SHALL create it automatically
6. WHEN a file already exists, THE Result_Exporter SHALL overwrite it unless the user specifies overwrite=False

### Requirement 8: 图形选项系统

**User Story:** As a user, I want to customize plot appearance, so that I can create publication-quality figures with consistent styling.

#### Acceptance Criteria

1. WHEN a user creates a PlotOptions instance, THE Plot_Options_Manager SHALL store default values for all plot parameters
2. WHEN a user calls set_plot_options, THE Plot_Options_Manager SHALL update the global default options
3. WHEN generating plots, THE Diagnostics_Plotter SHALL use the current PlotOptions settings
4. WHEN the user passes plot-specific options, THE Diagnostics_Plotter SHALL override the global defaults with the provided values
5. WHEN the user resets options, THE Plot_Options_Manager SHALL restore all settings to their original defaults
