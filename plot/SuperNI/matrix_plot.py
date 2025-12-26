import os
import json
import pandas as pd
import sys

# --- 用户配置 ---
# 请根据您的设置修改这些变量
METHOD = "clora-baseolora"
LR = "1e-05"
seq = 'order_4_llama'
# BASE_PATH = f"/home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/results/Longseq/{seq}/{METHOD}/outputs/{LR}/"
BASE_PATH = f"/home/qiuwenqi/LLM/Fedfinetune/FCL/{METHOD}/results/Longseq/{seq}/clora/outputs/{LR}/"
METRIC_PREFIX = "predict_exact_match_for_"
OUTPUT_EXCEL_FILE = f"cl_metrics_{seq}_{METHOD}_{LR}.xlsx"


# --- 配置结束 ---

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
        print(f"请检查您的 METHOD ('{METHOD}') 和 LR ('{LR}') 变量是否正确。")
        sys.exit(1)

    # 1. 发现并排序所有任务目录
    task_dirs = []
    for d in all_dirs:
        # 确保它是一个目录，并且以数字开头 (例如 '1-yelp')
        if os.path.isdir(os.path.join(base_path, d)) and d.split('-')[0].isdigit():
            task_dirs.append(d)

    # 关键：按数字前缀排序，确保任务顺序正确
    try:
        sorted_task_dirs = sorted(task_dirs, key=lambda x: int(x.split('-')[0]))
    except ValueError:
        print(f"错误: 无法对目录进行数字排序。找到: {task_dirs}")
        sys.exit(1)

    if not sorted_task_dirs:
        print(f"错误: 在 '{base_path}' 中未找到任务目录 (例如 '1-yelp', '2-amazon', ...)。")
        sys.exit(1)

    # 提取干净的任务名称列表 (['yelp', 'amazon', ...])
    task_names = [d.split('-', 1)[1] for d in sorted_task_dirs]
    num_tasks = len(task_names)

    # 2. 初始化数据结构 (Pandas DataFrame 非常适合此任务)
    # A[k, j] (索引 k, 列 j)
    df = pd.DataFrame(index=task_names, columns=task_names, dtype=float)

    # 3. 遍历每个任务 k (k = 0...N-1)，填充性能矩阵 A
    for k, task_dir in enumerate(sorted_task_dirs):
        # k 是行索引 (0-based)
        task_k_name = task_names[k]  # 当前任务的名称，例如 'yelp'
        json_file_path = os.path.join(base_path, task_dir, "all_results.json")

        if not os.path.exists(json_file_path):
            print(f"警告: 'all_results.json' 在 {task_dir} 中未找到。跳过第 {k + 1} 行。")
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
            task_j_name = task_names[j]  # 目标任务的名称
            metric_key = f"{METRIC_PREFIX}{task_j_name}"

            score = results_data.get(metric_key)

            if score is not None:
                df.loc[task_k_name, task_j_name] = score
            else:
                df.loc[task_k_name, task_j_name] = pd.NA
                print(f"警告: 在 {json_file_path} 中未找到键: {metric_key}")

    # --- 矩阵填充完毕，现在计算 AA 和 BWT ---

    print("--- 性能矩阵 (A_k,j) ---")
    print("A_k,j = 训练完任务 k 后，在任务 j 上的性能\n")

    aa_scores = []  # 存储每个 k 对应的 AA_k
    bwt_scores = []  # 存储每个 k 对应的 BWT_k

    # 5. 遍历每一行 (k) 来计算 AA_k 和 BWT_k
    for k in range(num_tasks):
        task_k_name = task_names[k]

        # --- 计算 AA_k (k=0...N-1) ---
        # 提取第 k 行，到第 k 列 (j=0...k)
        current_row_scores = df.loc[task_k_name].iloc[0:k + 1].values
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
            task_j_name = task_names[j]

            # a_k,j = 训练完任务 k 后，在任务 j 上的性能 (当前行, 过去列)
            a_k_j = df.loc[task_k_name, task_j_name]

            # a_j,j = 训练完任务 j 后，在任务 j 上的性能 (对角线)
            a_j_j = df.loc[task_j_name, task_j_name]

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
    pd.set_option('display.width', 1000)

    # 重命名索引以提高可读性
    df_display.index = [f"Task {i + 1} ({name})" for i, name in enumerate(task_names)]

    print(df_display.to_string(na_rep="-"))
    print(f"\n报告了 {num_tasks} 个任务。")
    print("AA_k = 对应行的简单平均值 (符合您图片中的公式(1))。")
    print("BWT_k = 对应行在 *过去* 任务上的性能与 *对角线* 性能之差的平均值 (符合公式(5))。")

    # --- *** 新增：保存到 Excel *** ---
    try:
        # 将 DataFrame 保存到 Excel 文件
        # na_rep='-' 确保 Excel 中的 'NaN' 也显示为 '-'
        # float_format='%.2f' 确保 Excel 中的数字格式正确
        df_display.to_excel(output_file, na_rep='-', float_format='%.2f')
        print(f"\n--- 成功！ ---")
        print(f"性能矩阵已保存到: {output_file}")
    except Exception as e:
        print(f"\n--- 错误 ---")
        print(f"无法将结果保存到 Excel 文件: {e}")
        print("请确保您已安装 'openpyxl' 库 (例如: pip install openpyxl)")


if __name__ == "__main__":
    # 将输出文件名传递给函数
    calculate_cl_metrics(BASE_PATH, OUTPUT_EXCEL_FILE)

