import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取 Excel 文件
excel_file = pd.ExcelFile('D:\\CodeLife\\CL\\FCL\\NewLLM\\results\\new_order_4\\forget_rate.xlsx')

# 获取指定工作表中的数据
df = excel_file.parse('Sheet1')

# 获取`遗忘率变化`列中数据作为方法名
methods = df['forget'].tolist()

# 获取`exact_match`列到`Unnamed: 14`列的数据作为 exactmatch 数据
exactmatch_data = df.loc[:, 'exact_match':'Unnamed: 14'].values.tolist()

# 将 exactmatch 数据转换为字典形式
exactmatch_data = {method: data for method, data in zip(methods, exactmatch_data)}

# 获取`rouge1`列到`Unnamed: 28`列的数据作为 rouge 数据
rouge_data = df.loc[:, 'rouge1':'Unnamed: 28'].values.tolist()

# 将 rouge 数据转换为字典形式
rouge_data = {method: data for method, data in zip(methods, rouge_data)}

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 绘制 exactmatch 遗忘率图
plt.figure(figsize=(12, 8))
for method in methods:
    data_length = len(exactmatch_data[method])
    if method.startswith('newAlg'):
        plt.plot(range(1, data_length + 1), exactmatch_data[method], marker='^', label=method, linestyle='--')
    else:
        plt.plot(range(1, data_length + 1), exactmatch_data[method], marker='o', label=method)
plt.xlabel('Tasks')
plt.ylabel('Forgetting Rate (exactmatch)')
plt.title('Forgetting Rate across Tasks (exactmatch)')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# 绘制 rouge 遗忘率图
plt.figure(figsize=(12, 8))
for method in methods:
    data_length = len(rouge_data[method])
    if method.startswith('newAlg'):
        plt.plot(range(1, data_length + 1), rouge_data[method], marker='^', label=method, linestyle='--')
    else:
        plt.plot(range(1, data_length + 1), rouge_data[method], marker='o', label=method)
plt.xlabel('Tasks')
plt.ylabel('Forgetting Rate (rouge)')
plt.title('Forgetting Rate across Tasks (rouge)')
plt.legend()
plt.xticks(rotation=45)
plt.show()