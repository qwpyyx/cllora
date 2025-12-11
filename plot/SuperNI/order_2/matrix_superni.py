import os
import json
import pandas as pd
import sys
import re

# --- 用户配置 ---
# 请根据您的设置修改这些变量
BENCHMARK = "SuperNI"
ORDER = "order_2_llama"
METHOD = "adaptive"
LR = "1e-04-budget11264"

# 基础路径，指向包含 '1-task748...', '2-task073...' 等的目录
# BASE_PATH = f"/home/qiuwenqi/LLM/Fedfinetune/FCL/{METHOD}/results/{BENCHMARK}/{ORDER}/{METHOD}/llama/outputs/{LR}/"
BASE_PATH = f"/home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/results/{BENCHMARK}/{ORDER}/{METHOD}/llama/outputs/{LR}/"
# /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/results/SuperNI/order_1_llama/lora_origin/llama/outputs/5e-05
# *** 新增：SuperNI 任务 ID 到指标的映射 ***
# (根据 image_3843e3.png)
SUPERNI_METRIC_MAP = {
    # Rouge-L
    'task639': 'Rouge-L',
    'task1590': 'Rouge-L',
    'task1729': 'Rouge-L',
    'task181': 'Rouge-L',
    'task748': 'Rouge-L',
    'task1510': 'Rouge-L',
    'task002': 'Rouge-L',
    'task073': 'Rouge-L',
    'task591': 'Rouge-L',
    'task511': 'Rouge-L',
    'task1290': 'Rouge-L',
    'task1572': 'Rouge-L',
    # accuracy
    'task363': 'accuracy',
    'task875': 'accuracy',
    'task1687': 'accuracy',
}

# *** 新增：将指标映射到 JSON 文件中的键前缀 ***
# (根据 all_results.json 示例)
METRIC_TO_JSON_KEY = {
    'accuracy': 'predict_exact_match_for_',
    'Rouge-L': 'predict_rougeL_for_'
}

# Excel 文件保存路径
OUTPUT_EXCEL_FILE = f"cl_metrics_{BENCHMARK}_{ORDER}_{METHOD}_{LR}.xlsx"


# --- 配置结束 ---


def get_task_id_from_name(folder_name):
    """从 '6-task1687_sentiment140_classification' 中提取 'task1687'"""
    parts = folder_name.split('-', 1)
    if len(parts) < 2:
        return None, None

    # full_key_name 是 'task1687_sentiment140_classification'
    full_key_name = parts[1]

    # task_id 是 'task1687'
    task_id_match = re.match(r"(task\d+)", full_key_name)
    if task_id_match:
        return full_key_name, task_id_match.group(1)
    return None, None


def calculate_cl_metrics(base_path, output_file):
    """
    扫描基础路径，读取所有任务结果，并计算AA和BWT矩阵。
    - AA_k (Average Accuracy): 训练到任务k时，在所有k个任务上的简单平均性能。
    - BWT_k (Backward Transfer): 训练到任务k时，在所有 k-1 个过去任务上的平均性能变化。
    """
    print(f"--- 正在计算持续学习指标 ---")
    print(f"扫描目录: {base_path}\n")

    try:
        all_dirs = os.listdir(base_path)
    except FileNotFoundError:
        print(f"错误: 目录未找到: {base_path}")
        print(f"请检查您的配置变量是否正确。")
        sys.exit(1)

    # 1. 发现并排序所有任务目录
    tasks_info = []  # 存储 (prefix, folder_name, full_key_name, task_id)
    for d in all_dirs:
        # 确保它是一个目录，并且以数字开头 (例如 '1-task748...')
        if os.path.isdir(os.path.join(base_path, d)) and d.split('-')[0].isdigit():
            prefix = int(d.split('-')[0])
            full_key_name, task_id = get_task_id_from_name(d)

            if task_id is None:
                print(f"警告: 无法从目录 {d} 中解析 task_id。跳过。")
                continue

            if task_id not in SUPERNI_METRIC_MAP:
                print(f"警告: 目录 {d} 中的 {task_id} 未在 SUPERNI_METRIC_MAP 中定义。跳过。")
                continue

            tasks_info.append((prefix, d, full_key_name, task_id))

    # 关键：按数字前缀排序
    tasks_info.sort(key=lambda x: x[0])

    if not tasks_info:
        print(f"错误: 在 '{base_path}' 中未找到有效的任务目录。")
        sys.exit(1)

    # 提取干净的 full_key_name 和 task_id 列表，保持排序
    task_full_key_names = [info[2] for info in tasks_info]
    task_ids = [info[3] for info in tasks_info]
    num_tasks = len(task_ids)

    # 2. 初始化数据结构 (索引和列都使用 full_key_name)
    df = pd.DataFrame(index=task_full_key_names, columns=task_full_key_names, dtype=float)

    # 3. 遍历每个任务 k (k = 0...N-1)，填充性能矩阵 A
    for k, (prefix_k, folder_k, full_key_k, task_id_k) in enumerate(tasks_info):

        json_file_path = os.path.join(base_path, folder_k, "all_results.json")

        if not os.path.exists(json_file_path):
            print(f"警告: 'all_results.json' 在 {folder_k} 中未找到。跳过第 {k + 1} 行。")
            continue

        try:
            with open(json_file_path, 'r') as f:
                results_data = json.load(f)
        except json.JSONDecodeError:
            print(f"错误: 无法解析 {json_file_path}。文件可能已损坏。")
            continue

        # 4. 填充矩阵的第 k 行：
        # 遍历所有 j <= k 的任务 (j = 0...k)
        for j in range(k + 1):
            full_key_j = task_full_key_names[j]  # 目标任务的 full_key_name
            task_id_j = task_ids[j]  # 目标任务的 task_id

            # --- 动态指标选择 ---
            # 1. 查找任务 j 需要什么指标 (accuracy or Rouge-L)
            metric_type = SUPERNI_METRIC_MAP[task_id_j]
            # 2. 查找该指标对应的 JSON 键前缀
            json_key_prefix = METRIC_TO_JSON_KEY[metric_type]
            # 3. 构造完整的 JSON 键
            metric_key = f"{json_key_prefix}{full_key_j}"
            # ------------------------

            score = results_data.get(metric_key)

            if score is not None:
                # df.loc[行k, 列j]
                df.loc[full_key_k, full_key_j] = score
            else:
                df.loc[full_key_k, full_key_j] = pd.NA
                print(f"警告: 在 {json_file_path} 中未找到键: {metric_key}")

    # --- 矩阵填充完毕，现在计算 AA 和 BWT ---

    print("--- 性能矩阵 (A_k,j) ---")
    print("A_k,j = 训练完任务 k 后，在任务 j 上的性能 (已根据任务自动选择Acc或RougeL)\n")

    aa_scores = []  # 存储每个 k 对应的 AA_k
    bwt_scores = []  # 存储每个 k 对应的 BWT_k

    # 5. 遍历每一行 (k) 来计算 AA_k 和 BWT_k
    for k in range(num_tasks):
        full_key_k = task_full_key_names[k]

        # --- 计算 AA_k (k=0...N-1) ---
        # 提取第 k 行，到第 k 列 (j=0...k)
        current_row_scores = df.loc[full_key_k].iloc[0:k + 1].values
        if pd.isna(current_row_scores).any() or len(current_row_scores) == 0:
            aa_k = pd.NA
        else:
            # 公式(1): AA_k = 1/(k+1) * Σ(a_k,j) [j=0..k]
            aa_k = sum(current_row_scores) / len(current_row_scores)
        aa_scores.append(aa_k)

        # --- 计算 BWT_k (k=0...N-1) ---
        # BWT_k 在 k=0 (第一个任务) 时未定义
        num_previous_tasks = k
        if num_previous_tasks == 0:
            bwt_scores.append(pd.NA)
            continue

        bwt_sum = 0.0
        valid_bwt_tasks = 0

        # 遍历所有过去的任务 j = 0...k-1
        for j in range(num_previous_tasks):
            full_key_j = task_full_key_names[j]

            # a_k,j = 训练完任务 k 后，在任务 j 上的性能 (当前行, 过去列)
            a_k_j = df.loc[full_key_k, full_key_j]

            # a_j,j = 训练完任务 j 后，在任务 j 上的性能 (对角线)
            a_j_j = df.loc[full_key_j, full_key_j]

            if pd.notna(a_k_j) and pd.notna(a_j_j):
                bwt_sum += (a_k_j - a_j_j)
                valid_bwt_tasks += 1

        # 公式(5): BWT_k = 1/k * Σ(a_k,j - a_j,j) [j=0..k-1]
        if valid_bwt_tasks > 0:
            bwt_k = bwt_sum / valid_bwt_tasks
            bwt_scores.append(bwt_k)
        else:
            bwt_scores.append(pd.NA)

    # 6. 格式化并打印最终的 DataFrame
    df_display = df.copy()
    # 将 AA_k 和 BWT_k 作为新列添加到末尾
    df_display["AA_k"] = aa_scores
    df_display["BWT_k"] = bwt_scores

    # 设置浮点数显示格式
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1200)  # 加宽

    # 重命名索引和列以提高可读性 (只显示 task_id)
    short_names = [f"T{i + 1}({name})" for i, name in enumerate(task_ids)]
    df_display.index = short_names
    df_display.columns = task_ids + ["AA_k", "BWT_k"]  # 列名只用 task_id

    print(df_display.to_string(na_rep="-"))
    print(f"\n报告了 {num_tasks} 个任务。")
    print("AA_k = 对应行的简单平均值 (符合您图片中的公式(1))。")
    print("BWT_k = 对应行在 *过去* 任务上的性能与 *对角线* 性能之差的平均值 (符合公式(5))。")

    # --- 保存到 Excel ---
    try:
        # 重命名索引以便在 Excel 中更清晰
        df_excel = df_display.copy()
        df_excel.index = [f"Task {i + 1} ({name})" for i, name in enumerate(task_full_key_names)]

        df_excel.to_excel(output_file, na_rep='-', float_format='%.2f')
        print(f"\n--- 成功！ ---")
        print(f"性能矩阵已保存到: {output_file}")
    except Exception as e:
        print(f"\n--- 错误 ---")
        print(f"无法将结果保存到 Excel 文件: {e}")
        print("请确保您已安装 'openpyxl' 库 (例如: pip install openpyxl)")


if __name__ == "__main__":
    calculate_cl_metrics(BASE_PATH, OUTPUT_EXCEL_FILE)