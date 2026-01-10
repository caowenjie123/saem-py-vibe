# Implementation Plan: SAEMIX Python 鲁棒性优化

## Overview

本实现计划将设计文档中的组件分解为可执行的编码任务。任务按依赖关系排序，确保每个任务都建立在前一个任务的基础上。测试任务与实现任务紧密配合，以便尽早发现问题。

## Tasks

- [x] 1. 设置测试基础设施
  - [x] 1.1 配置 hypothesis 测试框架
    - 在 conftest.py 中添加 hypothesis 配置文件
    - 设置 ci/dev/debug 三种配置
    - _Requirements: 8.1_
  - [x] 1.2 创建测试数据生成策略
    - 实现 valid_saemix_dataframe 策略
    - 实现 transformation_inputs 策略
    - _Requirements: 8.1_

- [x] 2. 实现 ID 转换助手函数
  - [x] 2.1 在 utils.py 中添加 id_to_index 和 index_to_id 函数
    - 实现 1-based 到 0-based 转换
    - 添加输入验证和错误处理
    - _Requirements: 7.4_
  - [x] 2.2 编写 ID 转换往返属性测试
    - **Property 13: ID Conversion Round-Trip**
    - **Validates: Requirements 7.4**

- [x] 3. Checkpoint - 确保基础设施测试通过
  - 运行 pytest tests/ -v，确保所有测试通过
  - 如有问题请询问用户

- [x] 4. 实现参数变换数值防护
  - [x] 4.1 修改 utils.py 中的 transphi 函数
    - 添加溢出检查
    - 添加 verbose 参数支持
    - _Requirements: 4.1, 4.3_
  - [x] 4.2 修改 utils.py 中的 transpsi 函数
    - 添加 LOG_EPS 和 LOGIT_EPS 常量
    - 实现 clip 逻辑防止 -Inf/+Inf
    - 添加最终有限性检查
    - _Requirements: 4.1, 4.2, 4.5_
  - [x] 4.3 编写变换输出有限性属性测试
    - **Property 7: Transformation Output Finiteness**
    - **Validates: Requirements 4.1, 4.2**
  - [x] 4.4 编写逆变换边界属性测试
    - **Property 9: Inverse Transformation Bounds**
    - **Validates: Requirements 4.5**

- [x] 5. 实现 SaemixData 辅助列处理修复
  - [x] 5.1 修改 data.py 中的 _process_data 方法
    - 构建 auxiliary_mapping 字典
    - 在列裁剪前添加辅助列到 all_cols
    - 添加列存在性验证
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  - [x] 5.2 编写辅助列映射保留属性测试
    - **Property 1: Auxiliary Column Mapping Preservation**
    - **Validates: Requirements 1.1**
  - [x] 5.3 编写辅助列错误处理属性测试
    - **Property 2: Auxiliary Column Error Handling**
    - **Validates: Requirements 1.2**

- [x] 6. 实现协变量校验逻辑修复
  - [x] 6.1 修改 data.py 中的 _validate_data 方法
    - 使用列表推导式替代遍历时修改
    - 分离 valid_covariates 和 ignored_covariates
    - 添加警告输出
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - [x] 6.2 编写协变量验证正确性属性测试
    - **Property 4: Covariate Validation Correctness**
    - **Validates: Requirements 2.1, 2.2**

- [x] 7. Checkpoint - 确保数据处理测试通过
  - 运行 pytest tests/test_data*.py -v
  - 如有问题请询问用户

- [x] 8. 实现统一随机数管理
  - [x] 8.1 修改 control.py 中的 saemix_control 函数
    - 添加 rng 参数
    - 创建 numpy.random.Generator 实例
    - 移除 np.random.seed() 调用
    - _Requirements: 3.1, 3.3, 3.4_
  - [x] 8.2 修改 algorithm/saem.py 传递 RNG
    - 从 control 获取 RNG
    - 传递给 estep 和 mstep
    - _Requirements: 3.2_
  - [x] 8.3 修改 algorithm/estep.py 使用 RNG
    - 替换 np.random.xxx() 为 rng.xxx()
    - _Requirements: 3.2_
  - [x] 8.4 修改 algorithm/conddist.py 使用 RNG
    - 替换 np.random.xxx() 为 rng.xxx()
    - _Requirements: 3.2_
  - [x] 8.5 修改 simulation.py 使用 RNG
    - 添加 rng 参数
    - 实现 RNG 优先级逻辑
    - _Requirements: 3.2_
  - [x] 8.6 编写 RNG 可复现性属性测试
    - **Property 5: RNG Reproducibility**
    - **Validates: Requirements 3.5**
  - [x] 8.7 编写全局 RNG 状态保留属性测试
    - **Property 6: Global RNG State Preservation**
    - **Validates: Requirements 3.3, 3.4, 3.6**

- [x] 9. Checkpoint - 确保 RNG 测试通过
  - 运行 pytest tests/test_rng*.py -v
  - 如有问题请询问用户

- [x] 10. 实现数值稳定性增强
  - [x] 10.1 在 algorithm/mstep.py 中添加 compute_omega_safe 函数
    - 实现特征值检查和修正
    - 添加 Cholesky 验证
    - _Requirements: 5.3, 5.4_
  - [x] 10.2 在 algorithm/likelihood.py 中添加 compute_log_likelihood_safe 函数
    - 添加 NaN/Inf 检查
    - 添加诊断信息
    - _Requirements: 5.2, 5.5_
  - [x] 10.3 编写协方差矩阵修正属性测试
    - **Property 12: Covariance Matrix Correction**
    - **Validates: Requirements 5.4**
  - [x] 10.4 编写对数似然错误处理属性测试
    - **Property 11: Log-Likelihood Error Handling**
    - **Validates: Requirements 5.2**

- [x] 11. 实现错误处理与依赖管理
  - [x] 11.1 修改 __init__.py 实现依赖检查
    - 核心依赖立即检查
    - 可选依赖延迟检查
    - 添加 _require_matplotlib 函数
    - _Requirements: 7.1, 7.2_
  - [x] 11.2 编写依赖缺失错误处理单元测试
    - 测试核心依赖缺失时的错误消息
    - 测试可选依赖缺失时的行为
    - _Requirements: 7.1, 7.2_

- [x] 12. Checkpoint - 确保数值稳定性测试通过
  - 运行 pytest tests/test_numerical*.py -v
  - 如有问题请询问用户

- [x] 13. 建立 CI 工作流
  - [x] 13.1 创建 .github/workflows/ci.yml
    - 配置多 Python 版本测试矩阵
    - 添加构建验证步骤
    - 添加 lint 检查
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 14. 建立回归测试
  - [x] 14.1 创建 tests/test_regression.py
    - 使用 theo.saemix.tab 数据集
    - 定义参考值和容差
    - 实现结果比较逻辑
    - _Requirements: 8.1, 8.2, 8.3, 8.5_

- [x] 15. Final Checkpoint - 完整测试套件
  - 运行 pytest tests/ -v --tb=short
  - 验证所有测试通过
  - 如有问题请询问用户

## Notes

- 所有任务都是必需的，包括测试任务
- 每个任务都引用了具体的需求以便追溯
- Checkpoint 任务确保增量验证
- 属性测试验证通用正确性属性
- 单元测试验证特定示例和边界情况
