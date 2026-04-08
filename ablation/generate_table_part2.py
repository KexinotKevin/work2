#!/usr/bin/env python3
"""
生成消融实验 Part 2（一致性阈值和池化策略）的汇总表格

用法: python generate_table_part2.py --exp_dir <实验根目录>

示例: 
  python generate_table_part2.py --exp_dir results/ablation/20260407_000752
  python generate_table_part2.py --exp_dir results/ablation/20260407_000752 --ours_exp results/hcd_all/20260406_122112/HCD/atlas_bna246__sc_fiber_count__fc_pcc_rest/split_70_15_15/seed_42

输出: 
  - ablation_tables/ablation_metrics_thr_pool_<timestamp>.csv: CSV格式表格
  - ablation_tables/ablation_metrics_thr_pool_<timestamp>.tex: LaTeX格式表格
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import argparse
import time

def concordance_correlation_coefficient(y_true, y_pred):
    """
    计算一致性相关系数（CCC, Concordance Correlation Coefficient）。
    CCC = 2 * cov(y_true, y_pred) / (var(y_true) + var(y_pred) + (mean(y_true) - mean(y_pred))^2)
    :param y_true: 真实标签，NumPy array
    :param y_pred: 预测标签，NumPy array
    :return: CCC值
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    mean_t = np.mean(y_true)
    mean_p = np.mean(y_pred)
    var_t = np.var(y_true)
    var_p = np.var(y_pred)
    
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    
    numerator = 2 * cov
    denominator = var_t + var_p + (mean_t - mean_p) ** 2
    
    if denominator == 0:
        return np.nan
        
    return numerator / denominator

def load_experiment_results(exp_path):
    """从实验结果路径加载 test.csv 数据，支持旧结构和新的嵌套结构"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    exp_full_path = os.path.normpath(os.path.join(project_root, exp_path))
    
    if not os.path.exists(exp_full_path):
        return []
    
    # 使用 glob 递归查找所有 label_* 目录下的 test.csv
    csv_files = glob.glob(os.path.join(exp_full_path, '**', 'label_*', 'test.csv'), recursive=True)
    
    results = []
    for test_csv in csv_files:
        label_dir = os.path.dirname(test_csv)
        df = pd.read_csv(test_csv)
        results.append({
            'label': os.path.basename(label_dir).replace("label_", ""),
            'rmse': df['repeat_rmse'].mean(),
            'mae': df['repeat_mae'].mean(),
            'r2': df['repeat_r2'].mean(),
            'pearson': df['pearson_corr'].mean(),
            'ccc': df['repeat_ccc'].mean()
        })
    
    return results

def format_metric(value):
    """格式化单个指标值"""
    if pd.isna(value):
        return "-"
    return f"{value:.4f}"

def generate_csv(all_results, experiments, timestamp):
    """
    生成与 ablation_metrics_thr_pool.csv 风格一致的 CSV 文件。
    格式：
    - 第1行: 标签名组（每个标签下4个指标位）
    - 第2行: Experiments, Settings, RMSE, MAE, $R^2$, Pearson $r$, ...
    - 数据行: 实验分组行 + 实验设置行 + 数据行
    """
    csv_out = f"ablation/ablation_tables/ablation_metrics_thr_pool_{timestamp}.csv"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    
    # 获取所有唯一的 Label 并排序
    all_labels = set()
    for exp_results in all_results.values():
        for r in exp_results:
            all_labels.add(r['label'])
    labels = sorted(all_labels)
    
    # 构建第1行：标签组名（简化显示）
    row1_cols = ["", ""]  # Experiments, Settings
    for label in labels:
        label_display = label.replace("nih_", "").replace("_unadjusted", "")
        row1_cols.extend([label_display, "", "", "", ""])
    
    # 构建第2行：指标名称
    row2_cols = ["Experiments", "Settings"]
    for _ in labels:
        row2_cols.extend(["RMSE", "MAE", "$R^2$", "Pearson $r$", "CCC"])
    
    # 构建数据行：每个实验设置一行，包含所有 label 的指标
    data_rows = []
    exp_groups = {
        "Threshold's Impact": ["Thresh (p=0.25)", "Thresh (p=0.50)", "Thresh (p=1.0)"],
        "Pooling Results' Impact": ["Pooling (GMP Only)", "Pooling (GAP Only)", "Ours"]
    }
    
    for group_name, exp_names in exp_groups.items():
        for exp_idx, exp_name in enumerate(exp_names):
            if exp_name in all_results and all_results[exp_name]:
                # 构建该实验设置的数据行
                row = [group_name if exp_idx == 0 else "", exp_name]
                for lbl in labels:
                    # 查找该实验在该 label 下的结果
                    label_result = None
                    for r in all_results[exp_name]:
                        if r['label'] == lbl:
                            label_result = r
                            break
                    
                    if label_result:
                        row.extend([
                            format_metric(label_result['rmse']),
                            format_metric(label_result['mae']),
                            format_metric(label_result['r2']),
                            format_metric(label_result['pearson']),
                            format_metric(label_result['ccc'])
                        ])
                    else:
                        row.extend(["", "", "", "", ""])
                data_rows.append(row)
    
    # 写入CSV
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        writer = pd.DataFrame([row1_cols, row2_cols] + data_rows)
        writer.to_csv(f, header=False, index=False)
    
    return csv_out

def generate_latex(all_results, experiments, timestamp):
    """
    生成与 ablation_metrics_thr_pool.csv 风格一致的 LaTeX 三线表。
    """
    csv_out = f"ablation/ablation_tables/ablation_metrics_thr_pool_{timestamp}.tex"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    
    # 获取所有唯一的 Label 并排序
    all_labels = set()
    for exp_results in all_results.values():
        for r in exp_results:
            all_labels.add(r['label'])
    labels = sorted(all_labels)
    
    # 动态计算列数: 2 (Experiments, Settings) + len(labels) * 5 (每个label 5个指标)
    n_metrics = 5  # RMSE, MAE, R^2, Pearson r, CCC
    col_spec = "ll" + "".join(["c" for _ in range(len(labels) * n_metrics)])
    
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Ablation Study: Consistency Threshold and Pooling Strategy}")
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    # 表头行1: 实验分组 + 设置 + 每个Label的列标题（简化显示）
    header_row1 = "\\textbf{Experiments} & \\textbf{Settings}"
    for label in labels:
        label_display = label.replace("nih_", "").replace("_unadjusted", "")
        header_row1 += f" & \\multicolumn{{{n_metrics}}}{{c}}{{\\textbf{{{label_display}}}}}"
    header_row1 += " \\\\"
    latex_lines.append(header_row1)
    
    # 表头行2: 指标名称
    header_row2 = "\\textbf{} & \\textbf{}"
    for _ in labels:
        header_row2 += " & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{R$^2$} & \\textbf{Pearson $r$} & \\textbf{CCC}"
    header_row2 = header_row2.strip() + " \\\\"
    latex_lines.append("\\midrule")
    latex_lines.append(header_row2)
    
    # 数据行：每个实验设置一行，包含所有 label 的指标
    exp_groups = {
        "Threshold's Impact": ["Thresh (p=0.25)", "Thresh (p=0.50)", "Thresh (p=1.0)"],
        "Pooling Results' Impact": ["Pooling (GMP Only)", "Pooling (GAP Only)", "Ours"]
    }
    
    for group_name, exp_names in exp_groups.items():
        for exp_idx, exp_name in enumerate(exp_names):
            if exp_name in all_results and all_results[exp_name]:
                # 构建该实验设置的数据行
                row_data = [group_name if exp_idx == 0 else "", exp_name]
                for lbl in labels:
                    # 查找该实验在该 label 下的结果
                    label_result = None
                    for r in all_results[exp_name]:
                        if r['label'] == lbl:
                            label_result = r
                            break
                    
                    if label_result:
                        row_data.extend([
                            format_metric(label_result['rmse']),
                            format_metric(label_result['mae']),
                            format_metric(label_result['r2']),
                            format_metric(label_result['pearson']),
                            format_metric(label_result['ccc'])
                        ])
                    else:
                        row_data.extend(["", "", "", "", ""])
                latex_lines.append(" & ".join(row_data) + " \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\label{tab:ablation_threshold_pooling}")
    latex_lines.append("\\end{table}")
    
    latex_content = "\n".join(latex_lines)
    
    with open(csv_out, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    return csv_out, latex_content

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成消融实验 Part 2 汇总表格 (阈值和池化策略)")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="消融实验根目录路径 (包含 thresh_*, pool_* 子目录)")
    parser.add_argument("--ours_exp", type=str, default=None,
                        help="Ours 实验数据目录路径 (可选，默认为 exp_dir/ours)")
    return parser.parse_args()

def main():
    print("=" * 80)
    print("Ablation Study Part 2: Consistency Threshold & Pooling Strategy")
    print("=" * 80)
    print()

    args = parse_args()
    
    # 实验配置：阈值和池化策略 (相对于 exp_dir)
    exp_dir = args.exp_dir
    ours_exp = args.ours_exp if args.ours_exp else os.path.join(exp_dir, "ours")
    
    experiments = {
        "Thresh (p=0.25)": os.path.join(exp_dir, "thresh_0.25"),
        "Thresh (p=0.50)": os.path.join(exp_dir, "thresh_0.50"),
        "Thresh (p=1.0)": os.path.join(exp_dir, "thresh_1.0"),
        "Pooling (GMP Only)": os.path.join(exp_dir, "pool_gmp"),
        "Pooling (GAP Only)": os.path.join(exp_dir, "pool_gap"),
        "Ours": ours_exp
    }
    
    # 收集所有实验结果
    all_results = {}
    for exp_name, exp_path in experiments.items():
        print(f"Loading results for: {exp_name}")
        results = load_experiment_results(exp_path)
        if results:
            all_results[exp_name] = results
            print(f"  -> Found {len(results)} label(s)")
        else:
            print(f"  -> No results found")
    
    if not all_results:
        print("\nNo results found. Exiting.")
        return
    
    print()
    print("=" * 80)
    
    # 生成时间戳用于文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存为 CSV
    csv_out = generate_csv(all_results, experiments, timestamp)
    print(f"[Success] Ablation summary Part 2 saved to {csv_out}")
    
    # 2. 生成并保存 LaTeX 表格
    tex_out, latex_content = generate_latex(all_results, experiments, timestamp)
    print(f"[Success] LaTeX table saved to {tex_out}")
    
    print()
    print("================== LaTeX Code for Paper ==================")
    print(latex_content)
    print("==========================================================")

if __name__ == "__main__":
    main()
