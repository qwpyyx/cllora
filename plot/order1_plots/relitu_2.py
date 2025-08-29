import os
import json
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_rel, wilcoxon

# -------------------- 1. 从文件里读取两个矩阵的相似度 --------------------
def load_task_vectors(base_dir, tasks, which='A'):
    vecs = []
    for task in tasks:
        adapter = os.path.join(base_dir, task, 'adapter')
        cfg = json.load(open(os.path.join(adapter, 'adapter_config.json')))
        r, r_sum = cfg['r'], cfg['r_sum']
        state = torch.load(os.path.join(adapter, 'adapter_model.bin'), map_location='cpu')
        mats = []
        suffix = 'lora_A.weight' if which=='A' else 'lora_B.weight'
        for k,v in state.items():
            if k.endswith(suffix):
                if which=='A':
                    mats.append(v[r_sum-r:r_sum, :])
                else:
                    mats.append(v[:, r_sum-r:r_sum])
        if not mats: continue
        mat = torch.cat(mats, dim=1 if which=='A' else 0)
        vecs.append(mat.flatten().numpy())
    return vecs

BASE_DIR_B = r'D:\CodeLife\CL\FCL\NewLLM\results\new_order_1\newAlg\tau0-5'
BASE_DIR_A = r'D:\CodeLife\CL\FCL\NewLLM\results\new_order_1\olora'
# TASKS = [f'{i}-{n}' for i,n in [
#     (1,'MNLI'),(2,'CB'),(3,'WiC'),(4,'COPA'),(5,'QQP'),
#     (6,'BoolQA'),(7,'RTE'),(8,'IMDB'),(9,'yelp'),(10,'amazon'),
#     (11,'SST-2'),(12,'dbpedia'),(13,'agnews'),(14,'MultiRC'),(15,'yahoo'),
# ]]
TASKS = [f'{i}-{n}' for i,n in [
    (1,'dbpedia'),(2,'amazon'),(3,'yahoo'),(4,'agnews'),
]]

vecs_A = load_task_vectors(BASE_DIR_A, TASKS, which='A')
vecs_B = load_task_vectors(BASE_DIR_B, TASKS, which='B')

sim_A = cosine_similarity(vecs_A)
sim_B = cosine_similarity(vecs_B)

def offdiag_values(sim):
    n = sim.shape[0]
    return sim[~np.eye(n, dtype=bool)]

off_A = offdiag_values(sim_A)
off_B = offdiag_values(sim_B)

# -------------------- 2. 计算全局指标 --------------------
mean_abs_A = np.mean(np.abs(off_A))
mean_abs_B = np.mean(np.abs(off_B))
max_abs_A  = np.max( np.abs(off_A))
max_abs_B  = np.max( np.abs(off_B))

print("A 矩阵 —— 平均绝对相似度：", f"{mean_abs_A:.3e}", "最大绝对相似度：", f"{max_abs_A:.3e}")
print("B 矩阵 —— 平均绝对相似度：", f"{mean_abs_B:.3e}", "最大绝对相似度：", f"{max_abs_B:.3e}")

# -------------------- 3. 显著性检验 --------------------
# 配对 t 检验
t_stat, p_val_t = ttest_rel(np.abs(off_A), np.abs(off_B))
# 非参数 Wilcoxon 符号秩检验
w_stat, p_val_w = wilcoxon(np.abs(off_A), np.abs(off_B))

print(f"配对 t-检验: t={t_stat:.3f}, p={p_val_t:.3e}")
print(f"Wilcoxon 符号秩检验: W={w_stat:.3f}, p={p_val_w:.3e}")
