import matplotlib.pyplot as plt
import numpy as np

# 方法名
methods = ["c-lora", "hydralora", "lora", "moelora",  "newAlg_tau0-5", "nlora", "olora"]
# x 轴任务
tasks = ["amazon", "yahoo", "agnews"]

# exactmatch 数据
exactmatch_data = {
    "c-lora": [-14.07, -8.24, -15.80],
    "hydralora": [-2.72, -3.16, -5.13],
    "lora": [-17.84, -4.59, -13.46],
    "moelora": [-3.07, -7.21, -5.27],
    "newAlg_tau0-5": [-19.68, -5.58, -7.20],
    "nlora": [-1.14, -1.73, -0.76],
    "olora": [-0.20, 0.42, 0.24]
}

# rouge 数据
rouge_data = {
    "c-lora": [-14.07, -8.82, -12.59],
    "hydralora": [-2.72, -2.33, -3.78],
    "lora": [-17.84, -3.23, -10.00],
    "moelora": [-3.07, -4.97, -4.67],
    "newAlg_tau0-5": [-19.68, -5.90, -7.43],
    "nlora": [-1.14, -1.81, -0.75],
    "olora": [-0.20, -1.74, -1.11]
}

# 绘制 exactmatch 遗忘率图
plt.figure(figsize=(8, 5))
for method in methods:
    plt.plot(tasks, exactmatch_data[method], marker='o', label=method)
plt.xlabel('Tasks')
plt.ylabel('Forgetting Rate (exactmatch)')
plt.title('Forgetting Rate across Tasks (exactmatch)')
plt.legend()
plt.show()

# 绘制 rouge 遗忘率图
plt.figure(figsize=(8, 5))
for method in methods:
    plt.plot(tasks, rouge_data[method], marker='o', label=method)
plt.xlabel('Tasks')
plt.ylabel('Forgetting Rate (rouge)')
plt.title('Forgetting Rate across Tasks (rouge)')
plt.legend()
plt.show()