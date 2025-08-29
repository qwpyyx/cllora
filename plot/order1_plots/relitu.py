import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# BASE_DIR = os.path.join('D:\\CodeLife\\CL\\FCL\\NewLLM\\CL-LoRA-LoRA\\new_order_1\\newalg\\tau0-5')
BASE_DIR = os.path.join('D:\\CodeLife\\CL\\FCL\\NewLLM\\results\\new_order_4\\newAlg_tau0-3_k1')
TASKS = [
    '1-MNLI',
    '2-CB',
    '3-WiC',
    '4-COPA',
    '5-QQP',
    '6-BoolQA',
    '7-RTE',
    '8-IMDB',
    '9-yelp',
    '10-amazon',
    '11-SST-2',
    '12-dbpedia',
    '13-agnews',
    '14-MultiRC',
    '15-yahoo'
]
# TASKS = ['1-dbpedia', '2-amazon', '3-yahoo', '4-agnews']
OUTPUT_DIR = os.path.join(BASE_DIR, 'loranew_B_similarity_order4')
os.makedirs(OUTPUT_DIR, exist_ok=True)

task_vectors = []  # 每个任务的向量表示
task_names = []     # 用于标注热图

for task in TASKS:
    adapter_dir = os.path.join(BASE_DIR, task, 'adapter')
    config_path = os.path.join(adapter_dir, 'adapter_config.json')
    model_path = os.path.join(adapter_dir, 'adapter_model.bin')
    if not (os.path.isfile(config_path) and os.path.isfile(model_path)):
        print(f'Skip {task}, adapter files not found')
        continue

    with open(config_path, 'r') as f:
        cfg = json.load(f)
    r = cfg.get('r', 0)
    r_sum = cfg.get('r_sum', 0)
    state = torch.load(model_path, map_location='cpu')

    new_B_list = []
    for k, v in state.items():
        if k.endswith('lora_B.weight'):
            new_B = v[:, r_sum - r : r_sum]  # 提取我们感兴趣的 B（从 r_sum - r 开始）
            new_B_list.append(new_B)

    if not new_B_list:
        print(f'No lora_B found for {task}')
        continue

    mat = torch.cat(new_B_list, dim=0)        # shape: [total_rows, r]
    vec = mat.flatten().numpy()               # 转为一维向量
    task_vectors.append(vec)
    task_names.append(task)

# 检查是否有足够的任务参与比较
if len(task_vectors) < 2:
    print("Not enough tasks for similarity comparison.")
    exit(0)

# 计算余弦相似度矩阵
sim_matrix = cosine_similarity(task_vectors)

# 可视化余弦相似度热图
plt.figure(figsize=(18, 16))
sns.heatmap(sim_matrix, annot=True, cmap='YlGnBu', xticklabels=task_names, yticklabels=task_names)
plt.title('Cosine Similarity between loranew_B of Different Tasks')
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, 'loranew_B_similarity_heatmap.png')
plt.savefig(out_path)
plt.close()
print(f'Saved similarity heatmap to {out_path}')

# # ----------------- A --------------------
# OUTPUT_DIR = os.path.join(BASE_DIR, 'loranew_A_similarity')
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# #
# task_vectors = []  # 每个任务的向量表示
# task_names = []     # 用于标注热图
#
# for task in TASKS:
#     adapter_dir = os.path.join(BASE_DIR, task, 'adapter')
#     config_path = os.path.join(adapter_dir, 'adapter_config.json')
#     model_path = os.path.join(adapter_dir, 'adapter_model.bin')
#     if not (os.path.isfile(config_path) and os.path.isfile(model_path)):
#         print(f'Skip {task}, adapter files not found')
#         continue
#
#     with open(config_path, 'r') as f:
#         cfg = json.load(f)
#     r = cfg.get('r', 0)
#     r_sum = cfg.get('r_sum', 0)
#     state = torch.load(model_path, map_location='cpu')
#
#     new_A_list = []
#     for k, v in state.items():
#         if k.endswith('lora_A.weight'):
#             new_A = v[r_sum - r: r_sum, :]
#             new_A_list.append(new_A)
#     if not new_A_list:
#         print(f'No lora_A found for {task}')
#         continue
#     mat = torch.cat(new_A_list, dim=1)
#     vec = mat.flatten().numpy()               # 转为一维向量
#     task_vectors.append(vec)
#     task_names.append(task)
#
# # 检查是否有足够的任务参与比较
# if len(task_vectors) < 2:
#     print("Not enough tasks for similarity comparison.")
#     exit(0)
#
# # 计算余弦相似度矩阵
# sim_matrix = cosine_similarity(task_vectors)
#
# # 可视化余弦相似度热图
# plt.figure(figsize=(18, 16))
# sns.heatmap(sim_matrix, annot=True, cmap='YlGnBu', xticklabels=task_names, yticklabels=task_names)
# plt.title('Cosine Similarity between loranew_A of Different Tasks')
# plt.tight_layout()
#
# out_path = os.path.join(OUTPUT_DIR, 'loranew_A_similarity_heatmap.png')
# plt.savefig(out_path)
# plt.close()
# print(f'Saved similarity heatmap to {out_path}')