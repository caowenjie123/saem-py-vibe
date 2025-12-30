import numpy as np
import pandas as pd
from saemix import saemix, saemix_data, saemix_model, saemix_control
import matplotlib.pyplot as plt


def linear_model(psi, id, xidep):
    """
    线性模型: y = a * x + b
    psi[id, 0] = a (斜率)
    psi[id, 1] = b (截距)
    """
    return psi[id, 0] * xidep[:, 0] + psi[id, 1]


np.random.seed(123)
n_subjects = 10
n_obs_per_subject = 5

true_pop_a = 2.5
true_pop_b = 1.0
omega_a = 0.5
omega_b = 0.3
sigma = 0.2

data_list = []
for i in range(n_subjects):
    subject_a = true_pop_a + np.random.normal(0, omega_a)
    subject_b = true_pop_b + np.random.normal(0, omega_b)
    
    x = np.linspace(0, 4, n_obs_per_subject)
    y = subject_a * x + subject_b + np.random.normal(0, sigma, n_obs_per_subject)
    
    for j in range(n_obs_per_subject):
        data_list.append({
            'Id': i + 1,
            'X': x[j],
            'Y': y[j]
        })

data = pd.DataFrame(data_list)

print("=" * 60)
print("SAEM参数估计演示")
print("=" * 60)
print(f"\n数据信息:")
print(f"  个体数量: {n_subjects}")
print(f"  每个个体观测数: {n_obs_per_subject}")
print(f"  总观测数: {len(data)}")
print(f"\n真实参数值:")
print(f"  总体均值 - a: {true_pop_a:.3f}, b: {true_pop_b:.3f}")
print(f"  随机效应标准差 - omega_a: {omega_a:.3f}, omega_b: {omega_b:.3f}")
print(f"  残差标准差: {sigma:.3f}")

saemix_data_obj = saemix_data(
    name_data=data,
    name_group='Id',
    name_predictors=['X'],
    name_response='Y',
    verbose=False
)

saemix_model_obj = saemix_model(
    model=linear_model,
    psi0=np.array([[2.0, 1.0]]),
    description="线性模型: y = a*x + b",
    name_modpar=["a", "b"],
    transform_par=[0, 0],
    covariance_model=np.array([[1, 0], [0, 1]]),
    error_init=[0.2, 0.0]
)

control = saemix_control(
    nbiter_saemix=(200, 100),
    nb_chains=2,
    nbiter_mcmc=(5, 5, 5, 1),
    display_progress=True,
    warnings=False
)

print("\n" + "=" * 60)
print("开始SAEM算法拟合...")
print("=" * 60)

result = saemix(
    model=saemix_model_obj,
    data=saemix_data_obj,
    control=control
)

print("\n" + "=" * 60)
print("参数估计结果")
print("=" * 60)

mean_phi = result.results.mean_phi
print(f"\n总体均值参数 (mean_phi):")
print(f"  a: {mean_phi[0, 0]:.4f} (真实值: {true_pop_a:.4f})")
print(f"  b: {mean_phi[0, 1]:.4f} (真实值: {true_pop_b:.4f})")

if result.results.omega is not None:
    omega = result.results.omega
    print(f"\n随机效应协方差矩阵 (omega):")
    print(f"  omega_a: {np.sqrt(omega[0, 0]):.4f} (真实值: {omega_a:.4f})")
    print(f"  omega_b: {np.sqrt(omega[1, 1]):.4f} (真实值: {omega_b:.4f})")
    print(f"  协方差矩阵:\n{omega}")

if result.results.respar is not None:
    print(f"\n残差参数 (respar):")
    print(f"  sigma: {result.results.respar[0]:.4f} (真实值: {sigma:.4f})")

print_phi = result.results.cond_mean_phi
if print_phi is not None and print_phi.shape[0] > 0:
    print(f"\n个体参数估计 (前5个个体):")
    for i in range(min(5, print_phi.shape[0])):
        print(f"  个体 {i+1}: a={print_phi[i, 0]:.4f}, b={print_phi[i, 1]:.4f}")

print("\n" + "=" * 60)
print("预测结果")
print("=" * 60)

ppred = result.predict(type="ppred")
ipred = result.predict(type="ipred")

print(f"\n群体预测 (ppred) - 前10个值:")
print(ppred[:10])

print(f"\n个体预测 (ipred) - 前10个值:")
print(ipred[:10])

mse_ppred = np.mean((data['Y'].values - ppred) ** 2)
mse_ipred = np.mean((data['Y'].values - ipred) ** 2)
print(f"\n预测误差:")
print(f"  ppred MSE: {mse_ppred:.4f}")
print(f"  ipred MSE: {mse_ipred:.4f}")

print("\n" + "=" * 60)
print("演示完成")
print("=" * 60)
