import re
from pathlib import Path
import numpy as np
import pandas as pd

LLAMA_ROOT = Path(r"C:\Users\Wenqi\Desktop\icml\所有数据\主表\表1")
T5_ROOT    = Path(r"C:\Users\Wenqi\Desktop\icml\所有数据\主表\表2")   # 改成你的 T5-Large 路径

# 如果 T5 文件名里不是 "_t5_"，比如是 "_t5large_"，改这里
BACKBONES = {
    "LLaMA2-7B": {"root": LLAMA_ROOT, "keyword": "llama"},
    "T5-Large":  {"root": T5_ROOT,    "keyword": "t5"},
}

# 方法名映射，可按你的论文命名习惯改
METHOD_NAME_MAP = {
    "clora": "C-LoRA",
    "ewc": "EWC",
    "gem": "A-GEM",
    "hydralora": "HydraLoRA",
    "lora_origin": "LoRA",
    "lorm": "LoRM",
    "nlora": "N-LoRA",
    "olora": "O-LoRA",
    "pilora": "PILoRA",
    "replay": "Replay",
    "RieSelect": "RieSelect",
}

# 最终表格显示顺序
METHOD_ORDER = [
    "LoRA",
    "O-LoRA",
    "N-LoRA",
    "HydraLoRA",
    "C-LoRA",
    "EWC",
    "Replay",
    "GEM",
    "LoRM",
    "PiLoRA",
    "RieSelect",
]


def safe_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip()
        if x in {"", "-"}:
            return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def parse_method_from_filename(file_path: Path, backbone_keyword: str):
    stem = file_path.stem
    parts = stem.split("_")

    try:
        idx = parts.index(backbone_keyword)
    except ValueError:
        raise ValueError(f"Cannot find backbone keyword '{backbone_keyword}' in filename: {file_path.name}")

    method_raw = "_".join(parts[idx + 1 : -1])
    return method_raw


def load_one_excel_metrics(xlsx_path: Path):
    """
    从一个 AA 矩阵 Excel 中提取：
    1) Mean A_jj：对角线平均（平均 new-task / immediate-task accuracy）
    2) Mean A_Tj：最后一行前 T-1 列平均（最终 old-task retained accuracy）
    3) Final BWT：最后一行 BWT_k
    """
    df = pd.read_excel(xlsx_path)

    # 除去第一列（任务描述列）和指标列，其余都视为任务列
    exclude_cols = {"AA_k", "BWT_k"}
    first_col = df.columns[0]
    task_cols = [c for c in df.columns if c not in exclude_cols and c != first_col]

    if not task_cols:
        raise ValueError(f"No task columns found in {xlsx_path}")

    T = len(task_cols)
    if len(df) < T:
        raise ValueError(f"Row count < number of task columns in {xlsx_path}")

    task_mat = df[task_cols].applymap(safe_float).to_numpy(dtype=float)

    # Mean A_jj：对角线平均
    diag_vals = []
    for i in range(min(T, task_mat.shape[0])):
        diag_vals.append(task_mat[i, i])
    mean_A_jj = float(np.nanmean(diag_vals))

    # Mean A_Tj：最后一行前 T-1 个任务的平均
    final_row = task_mat[T - 1, :]
    mean_A_Tj = float(np.nanmean(final_row[:T - 1]))

    # Final BWT
    final_bwt = safe_float(df.iloc[T - 1]["BWT_k"])

    return {
        "mean_A_jj": mean_A_jj,
        "mean_A_Tj": mean_A_Tj,
        "final_bwt": final_bwt,
    }


def collect_backbone_results(root: Path, keyword: str):
    results = {}

    order_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("order")])
    if not order_dirs:
        raise FileNotFoundError(f"No order folders found under: {root}")

    for order_dir in order_dirs:
        # 不再假设文件名一定带 SuperNI
        files = sorted([fp for fp in order_dir.glob("*.xlsx") if f"_{keyword}_" in fp.stem])

        if not files:
            print(f"[WARN] No matched files in {order_dir} for keyword '{keyword}'")
            continue

        for fp in files:
            try:
                method_raw = parse_method_from_filename(fp, keyword)
                method_name = METHOD_NAME_MAP.get(method_raw, method_raw)

                metrics = load_one_excel_metrics(fp)
                metrics["order"] = order_dir.name
                metrics["file"] = fp.name

                results.setdefault(method_name, []).append(metrics)

            except Exception as e:
                print(f"[WARN] Skip {fp.name}: {e}")

    return results


def aggregate_results(results_dict):
    agg = {}
    for method, rows in results_dict.items():
        if not rows:
            continue

        agg[method] = {
            "mean_A_jj": float(np.mean([r["mean_A_jj"] for r in rows])),
            "mean_A_Tj": float(np.mean([r["mean_A_Tj"] for r in rows])),
            "mean_BWT": float(np.mean([r["final_bwt"] for r in rows])),
            "num_orders": len(rows),
        }
    return agg


def build_final_table(all_backbone_aggs):
    all_methods = set()
    for backbone_name, agg in all_backbone_aggs.items():
        all_methods.update(agg.keys())

    ordered_methods = [m for m in METHOD_ORDER if m in all_methods]
    for m in sorted(all_methods):
        if m not in ordered_methods:
            ordered_methods.append(m)

    rows = []
    for method in ordered_methods:
        row = {"Method": method}
        for backbone_name in BACKBONES.keys():
            agg = all_backbone_aggs.get(backbone_name, {})
            if method in agg:
                row[f"{backbone_name} Mean A_jj"] = agg[method]["mean_A_jj"]
                row[f"{backbone_name} Mean A_Tj"] = agg[method]["mean_A_Tj"]
                row[f"{backbone_name} Mean BWT"] = agg[method]["mean_BWT"]
            else:
                row[f"{backbone_name} Mean A_jj"] = np.nan
                row[f"{backbone_name} Mean A_Tj"] = np.nan
                row[f"{backbone_name} Mean BWT"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def to_latex_table(df: pd.DataFrame, caption: str, label: str):
    fmt_df = df.copy()
    for c in fmt_df.columns:
        if c != "Method":
            fmt_df[c] = fmt_df[c].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")

    latex = fmt_df.to_latex(
        index=False,
        escape=False,
        column_format="l" + "c" * (len(fmt_df.columns) - 1),
        caption=caption,
        label=label,
    )
    return latex


def main():
    all_backbone_aggs = {}

    for backbone_name, cfg in BACKBONES.items():
        print(f"\n===== Processing {backbone_name} =====")
        results = collect_backbone_results(cfg["root"], cfg["keyword"])
        agg = aggregate_results(results)
        all_backbone_aggs[backbone_name] = agg

        for method, vals in agg.items():
            print(
                f"{method:12s} | orders={vals['num_orders']} | "
                f"Mean A_jj={vals['mean_A_jj']:.2f} | "
                f"Mean A_Tj={vals['mean_A_Tj']:.2f} | "
                f"Mean BWT={vals['mean_BWT']:.2f}"
            )

    final_df = build_final_table(all_backbone_aggs)

    # 统一保留两位小数
    for c in final_df.columns:
        if c != "Method":
            final_df[c] = final_df[c].round(2)

    out_csv = Path("table_R2_Q4.csv")
    final_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    caption = (
        "Secondary analysis based on the AA matrices from Table 1 (LLaMA2-7B) and "
        "Table 6 (T5-Large). Mean $A_{j,j}$ denotes mean immediate new-task accuracy; "
        "mean $A_{T,j}$ denotes mean final retained old-task accuracy on earlier tasks; "
        "mean BWT is averaged over the reported orders."
    )
    label = "tab:r2_q4"
    latex_table = to_latex_table(final_df, caption=caption, label=label)

    out_tex = Path("table_R2_Q4.tex")
    out_tex.write_text(latex_table, encoding="utf-8")

    print("\n===== Final Table R2-Q4 =====")
    print(final_df.to_string(index=False))

    print("\n===== LaTeX =====")
    print(latex_table)

    print(f"\nSaved CSV: {out_csv.resolve()}")
    print(f"Saved TEX: {out_tex.resolve()}")


if __name__ == "__main__":
    main()