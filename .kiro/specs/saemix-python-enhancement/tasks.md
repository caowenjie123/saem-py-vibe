# Implementation Plan: Python saemix Enhancement

## Overview

本实现计划将 Python saemix 库的功能与 R 版本对齐，按照优先级分阶段实现。每个任务都是独立的编码单元，可以增量完成并验证。

## Tasks

- [x] 1. Phase 1: 核心功能完善

- [x] 1.1 实现条件分布估计模块
  - 创建 `saemix/algorithm/conddist.py`
  - 实现 Metropolis-Hastings MCMC 采样算法
  - 计算条件均值、条件方差、收缩估计
  - 支持多链采样和收敛诊断
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.7_

- [x] 1.2 编写条件分布估计的属性测试
  - **Property 1: MCMC Conditional Distribution Output Structure**
  - **Property 2: Shrinkage Bounds**
  - **Property 3: Sample Count Consistency**
  - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.7**

- [x] 1.3 实现模型比较模块
  - 创建 `saemix/compare.py`
  - 实现 `compare_saemix` 函数
  - 计算 AIC, BIC, BIC_cov
  - 支持多种似然计算方法 (is, gq)
  - 返回比较结果 DataFrame
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 1.4 编写模型比较的属性测试
  - **Property 4: Model Comparison Output Correctness**
  - **Validates: Requirements 2.1, 2.2, 2.3, 2.5**

- [x] 1.5 完善结果对象 SaemixRes
  - 修改 `saemix/results.py`
  - 添加置信区间计算 (`compute_confidence_intervals`)
  - 添加迭代历史记录 (`parpop`, `allpar`)
  - 添加预测值 DataFrame (`predictions`)
  - 添加残差计算 (`ires`, `wres`, `pd_`)
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 1.6 编写结果对象的属性测试
  - **Property 5: Confidence Interval Computation**
  - **Property 6: Iteration History Recording**
  - **Property 7: Predictions and Residuals Structure**
  - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

- [x] 1.7 更新 SAEM 算法以记录迭代历史
  - 修改 `saemix/algorithm/saem.py`
  - 在每次迭代后记录参数值到 `parpop` 和 `allpar`
  - 确保与现有功能兼容
  - _Requirements: 3.2, 3.3_

- [x] 1.8 Checkpoint - Phase 1 验证
  - 确保所有测试通过
  - 使用 theo.saemix.tab 数据验证功能
  - 如有问题请询问用户

- [ ] 2. Phase 2: 模型选择功能

- [ ] 2.1 实现逐步回归模块
  - 创建 `saemix/stepwise.py`
  - 实现 `forward_procedure` 前向选择
  - 实现 `backward_procedure` 后向消除
  - 实现 `stepwise_procedure` 双向逐步
  - 基于 BIC 进行协变量选择
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

- [ ] 2.2 编写逐步回归的属性测试
  - **Property 8: Stepwise Selection Optimality**
  - **Validates: Requirements 4.1, 4.2, 4.3, 4.5**

- [ ] 2.3 实现模拟功能模块
  - 创建 `saemix/simulation.py`
  - 实现 `simulate_saemix` 连续响应模拟
  - 实现 `simulate_discrete_saemix` 离散响应模拟
  - 支持不确定性传播
  - 返回包含模拟数据的 DataFrame
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

- [ ] 2.4 编写模拟功能的属性测试
  - **Property 9: Simulation Reproducibility**
  - **Property 10: Simulation Output Structure**
  - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.7**

- [ ] 2.5 Checkpoint - Phase 2 验证
  - 确保所有测试通过
  - 验证逐步回归与 R 版本结果一致
  - 如有问题请询问用户

- [ ] 3. Phase 3: 诊断功能增强

- [ ] 3.1 实现收敛诊断图
  - 修改 `saemix/diagnostics.py`
  - 实现 `plot_convergence` 参数收敛图
  - 实现 `plot_likelihood` 似然轨迹图
  - 使用迭代历史数据绘图
  - _Requirements: 6.1, 6.2_

- [ ] 3.2 实现参数-协变量关系图
  - 实现 `plot_parameters_vs_covariates` 参数 vs 协变量图
  - 实现 `plot_randeff_vs_covariates` 随机效应 vs 协变量图
  - 支持连续和分类协变量
  - _Requirements: 6.3, 6.4_

- [ ] 3.3 实现参数分布图
  - 实现 `plot_marginal_distribution` 边际分布图
  - 实现 `plot_correlations` 相关性矩阵图
  - 支持自定义参数选择
  - _Requirements: 6.5, 6.6_

- [ ] 3.4 编写诊断图的单元测试
  - 测试图形函数返回正确的 Figure 对象
  - 测试图形尺寸选项生效
  - **Validates: Requirements 6.7**

- [ ] 3.5 Checkpoint - Phase 3 验证
  - 确保所有诊断图功能正常
  - 视觉检查图形输出
  - 如有问题请询问用户

- [ ] 4. Phase 4: 辅助功能

- [ ] 4.1 实现结果导出模块
  - 创建 `saemix/export.py`
  - 实现 `save_results` 保存所有结果
  - 实现 `export_to_csv` 导出指定内容
  - 实现 `save_plots` 保存诊断图
  - 支持目录自动创建
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [ ] 4.2 编写结果导出的属性测试
  - **Property 11: File Export Round-Trip**
  - **Validates: Requirements 7.1, 7.2, 7.3, 7.4**

- [ ] 4.3 实现图形选项系统
  - 创建 `saemix/plot_options.py`
  - 实现 `PlotOptions` 数据类
  - 实现 `set_plot_options`, `get_plot_options`, `reset_plot_options`
  - 集成到所有诊断图函数
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 4.4 编写图形选项的属性测试
  - **Property 12: Plot Options Management**
  - **Validates: Requirements 8.2, 8.3, 8.4, 8.5**

- [ ] 4.5 更新模块导出
  - 修改 `saemix/__init__.py`
  - 导出所有新增的公共函数和类
  - 更新文档字符串
  - _Requirements: All_

- [ ] 4.6 Checkpoint - Phase 4 验证
  - 确保所有测试通过
  - 验证导出文件可以正确读取
  - 如有问题请询问用户

- [ ] 5. 集成与文档

- [ ] 5.1 编写集成测试
  - 创建 `tests/test_integration_enhanced.py`
  - 测试完整工作流：数据加载 → 拟合 → 条件分布 → 模型比较 → 导出
  - 使用 theo.saemix.tab 和 cow.saemix.tab 数据
  - _Requirements: All_

- [ ] 5.2 更新示例代码
  - 修改 `examples/basic_example.py`
  - 添加新功能的使用示例
  - 添加条件分布估计示例
  - 添加模型比较示例
  - _Requirements: All_

- [ ] 5.3 Final Checkpoint - 完整验证
  - 运行所有测试确保通过
  - 验证与 R 版本结果一致性
  - 如有问题请询问用户

## Notes

- All tasks are required for complete implementation
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- 建议按 Phase 顺序实现，每个 Phase 完成后进行验证
- 使用 `saemix-main/data/` 中的数据进行测试
- 注意 Python 0-based 索引与 R 1-based 索引的差异
