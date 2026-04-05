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

def load_experiment_results(exp_path):
    """从实验结果路径加载 test.csv 数据"""
    # 获取脚本所在目录的父目录（项目根目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 构建完整路径
    exp_full_path = os.path.normpath(os.path.join(project_root, exp_path))
    
    # 查找 label 目录（直接在 exp_full_path 下）
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
    """生成 LaTeX 表格，按 Label 分组展示"""
    if not all_results:
        return "No results available."
    
    # 获取所有唯一的 Label
    all_labels = set()
    for exp_results in all_results.values():
        for r in exp_results:
            all_labels.add(r['label'])
    labels = sorted(all_labels)
    
    # 表头：每个实验一行，每个 Label 下显示指标
    header = """\\begin{table}[htbp]
\\centering
\\caption{Ablation Study: Consistency Threshold and Pooling Strategy}
\\begin{tabular}{l|c|c|c|c}
\\hline
\\textbf{Experiment} & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{R$^2$} & \\textbf{Pearson r} \\\\"""
    
    rows = []
    
    # 按实验顺序输出
    for exp_name in all_results.keys():
        if exp_name not in all_results or not all_results[exp_name]:
            continue
            
        results = all_results[exp_name]
        
        # 按 Label 分组输出，每个 Label 一行
        for r in results:
            label = r['label']
            exp_display = f"{exp_name}" if r == results[0] else ""  # 只在第一行显示实验名
            row = f"\\hline\n{exp_display} ({label}) & {format_metric(r['rmse_mean'], r['rmse_std'])} & {format_metric(r['mae_mean'], r['mae_std'])} & {format_metric(r['r2_mean'], r['r2_std'])} & {format_metric(r['pearson_mean'], r['pearson_std'])} \\\\"
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

    # 实验配置：阈值和池化策略 (路径相对于项目根目录)
    timestamp = sys.argv[1]
    experiments = {
        "Thresh (p=0.25)": f"results/ablation/{timestamp}/thresh_0.25",
        "Thresh (p=0.50)": f"results/ablation/{timestamp}/thresh_0.50",
        # "Ours (Thresh p=0.75, Concat)": f"results/ablation/{timestamp}/ours",
        "Thresh (p=1.0)": f"results/ablation/{timestamp}/thresh_1.0",
        "Pooling (GMP Only)": f"results/ablation/{timestamp}/pool_gmp",
        "Pooling (GAP Only)": f"results/ablation/{timestamp}/pool_gap"
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
    
    md_header = "| Experiment (Label) | RMSE | MAE | R² | Pearson r |"
    md_sep = "|--------------------|------|-----|----|-----------|"
    
    md_rows = []
    results_for_csv = []
    for exp_name, exp_path in experiments.items():
        results = load_experiment_results(exp_path)
        if results:
            for r in results:
                label = r['label']
                md_rows.append(f"| {exp_name} ({label}) | {r['rmse_mean']:.4f}±{r['rmse_std']:.4f} | {r['mae_mean']:.4f}±{r['mae_std']:.4f} | {r['r2_mean']:.4f}±{r['r2_std']:.4f} | {r['pearson_mean']:.4f}±{r['pearson_std']:.4f} |")
                results_for_csv.append({
                    "Experiment": exp_name,
                    "Label": label,
                    "RMSE_mean": r['rmse_mean'],
                    "RMSE_std": r['rmse_std'],
                    "MAE_mean": r['mae_mean'],
                    "MAE_std": r['mae_std'],
                    "R2_mean": r['r2_mean'],
                    "R2_std": r['r2_std'],
                    "Pearson_mean": r['pearson_mean'],
                    "Pearson_std": r['pearson_std']
                })
    
    print(md_header)
    print(md_sep)
    for row in md_rows:
        print(row)
    
    # 保存聚合结果为 CSV
    if results_for_csv:
        df_summary = pd.DataFrame(results_for_csv)
        csv_out = f"ablation_metrics_summary_{timestamp}.csv"
        df_summary.to_csv(csv_out, index=False)
        print(f"\n[Success] Ablation summary Part 2 saved to {csv_out}")

if __name__ == "__main__":
    main()