#!/usr/bin/env python3
"""
生成消融实验 Part 2（一致性阈值和池化策略）的汇总表格

用法: python generate_table_part2.py

输出: 打印 LaTeX 表格格式的结果
"""

import os
import sys
import pandas as pd
import numpy as np

# 实验配置：阈值和池化策略
timestamp = sys.argv[1]
experiments = {
    "Thresh (p=0.25)": f"./results/{timestamp}/thresh_0.25",
    "Thresh (p=0.50)": f"./results/{timestamp}/thresh_0.50",
    "Ours (Thresh p=0.75, Concat)": f"./results/{timestamp}/ours",
    "Thresh (p=1.0)": f"./results/{timestamp}/thresh_1.0",
    "Pooling (GMP Only)": f"./results/{timestamp}/pool_gmp",
    "Pooling (GAP Only)": f"./results/{timestamp}/pool_gap"
}

def load_experiment_results(exp_path):
    """从实验结果路径加载 test.csv 数据"""
    # 查找结果目录 (timestamp-based folder)
    exp_full_path = os.path.join(os.path.dirname(__file__), exp_path.lstrip('./'))
    
    # 查找最新的 timestamp 目录
    if os.path.exists(exp_full_path):
        subdirs = [d for d in os.listdir(exp_full_path) if os.path.isdir(os.path.join(exp_full_path, d))]
        if subdirs:
            # 使用最新的目录
            latest_dir = sorted(subdirs)[-1]
            exp_full_path = os.path.join(exp_full_path, latest_dir)
    
    # 查找 label 目录
    label_dirs = [d for d in os.listdir(exp_full_path) if d.startswith("label_")] if os.path.exists(exp_full_path) else []
    
    results = []
    for label_dir in label_dirs:
        test_csv = os.path.join(exp_full_path, label_dir, "test.csv")
        if os.path.exists(test_csv):
            df = pd.read_csv(test_csv)
            results.append({
                'label': label_dir.replace("label_", ""),
                'rmse_mean': df['repeat_rmse'].mean(),
                'rmse_std': df['repeat_rmse'].std(),
                'mae_mean': df['repeat_mae'].mean(),
                'mae_std': df['repeat_mae'].std(),
                'r2_mean': df['repeat_r2'].mean(),
                'r2_std': df['repeat_r2'].std(),
                'pearson_mean': df['pearson_corr'].mean(),
                'pearson_std': df['pearson_corr'].std()
            })
    
    return results

def format_metric(value, std):
    """格式化指标为均值 ± 标准差"""
    return f"${value:.4f} \\pm {std:.4f}$"

def generate_latex_table(all_results):
    """生成 LaTeX 表格"""
    # 表头
    header = """\\begin{table}[htbp]
\\centering
\\caption{Ablation Study: Consistency Threshold and Pooling Strategy}
\\begin{tabular}{l|c|c|c|c}
\\hline
\\textbf{Experiment} & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{R$^2$} & \\textbf{Pearson r} \\\\"""
    
    rows = []
    for exp_name, exp_path in experiments.items():
        results = load_experiment_results(exp_path)
        if results:
            # 汇总所有 label 的结果
            avg_rmse = np.mean([r['rmse_mean'] for r in results])
            avg_rmse_std = np.mean([r['rmse_std'] for r in results])
            avg_mae = np.mean([r['mae_mean'] for r in results])
            avg_mae_std = np.mean([r['mae_std'] for r in results])
            avg_r2 = np.mean([r['r2_mean'] for r in results])
            avg_r2_std = np.mean([r['r2_std'] for r in results])
            avg_pearson = np.mean([r['pearson_mean'] for r in results])
            avg_pearson_std = np.mean([r['pearson_std'] for r in results])
            
            row = f"\\hline\n{exp_name} & {format_metric(avg_rmse, avg_rmse_std)} & {format_metric(avg_mae, avg_mae_std)} & {format_metric(avg_r2, avg_r2_std)} & {format_metric(avg_pearson, avg_pearson_std)} \\\\"
            rows.append(row)
    
    footer = """\\hline
\\end{tabular}
\\label{tab:ablation_threshold_pooling}
\\end{table}"""
    
    return header + "\n" + "\n".join(rows) + "\n" + footer

def main():
    print("=" * 80)
    print("Ablation Study Part 2: Consistency Threshold & Pooling Strategy")
    print("=" * 80)
    print()
    
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
    
    print()
    print("=" * 80)
    print("Generating LaTeX Table...")
    print("=" * 80)
    print()
    
    latex_table = generate_latex_table(all_results)
    print(latex_table)
    
    # 同时打印 Markdown 表格便于查看
    print()
    print("=" * 80)
    print("Markdown Table (for quick reference)")
    print("=" * 80)
    print()
    
    md_header = "| Experiment | RMSE | MAE | R² | Pearson r |"
    md_sep = "|------------|------|-----|----|-----------|"
    
    md_rows = []
    for exp_name, exp_path in experiments.items():
        results = load_experiment_results(exp_path)
        if results:
            avg_rmse = np.mean([r['rmse_mean'] for r in results])
            avg_rmse_std = np.mean([r['rmse_std'] for r in results])
            avg_mae = np.mean([r['mae_mean'] for r in results])
            avg_mae_std = np.mean([r['mae_std'] for r in results])
            avg_r2 = np.mean([r['r2_mean'] for r in results])
            avg_r2_std = np.mean([r['r2_std'] for r in results])
            avg_pearson = np.mean([r['pearson_mean'] for r in results])
            avg_pearson_std = np.mean([r['pearson_std'] for r in results])
            
            md_rows.append(f"| {exp_name} | {avg_rmse:.4f}±{avg_rmse_std:.4f} | {avg_mae:.4f}±{avg_mae_std:.4f} | {avg_r2:.4f}±{avg_r2_std:.4f} | {avg_pearson:.4f}±{avg_pearson_std:.4f} |")
    
    print(md_header)
    print(md_sep)
    for row in md_rows:
        print(row)

if __name__ == "__main__":
    main()