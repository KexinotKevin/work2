import os
import sys
import glob
import pandas as pd
import numpy as np

def find_test_csvs_with_label(base_dir):
    """在指定的消融变体目录下，找到所有 label_* 子目录下的 test.csv，返回 (label_name, csv_path) 列表"""
    # 先查找所有 label_* 目录
    label_dirs = []
    if os.path.exists(base_dir):
        for d in os.listdir(base_dir):
            if d.startswith("label_"):
                label_dirs.append(d)
    
    results = []
    for label_dir in sorted(label_dirs):
        csv_path = os.path.join(base_dir, label_dir, "test.csv")
        if os.path.exists(csv_path):
            label_name = label_dir.replace("label_", "")
            results.append((label_name, csv_path))
    
    return results

def main():
    # 定义四个消融实验及其对应的文件夹
    timestamp = sys.argv[1]
    experiments = {
        "Ours": f"../results/ablation/{timestamp}/ours",
        "w/o GRL": f"../results/ablation/{timestamp}/wo_grl",
        "w/o PR": f"../results/ablation/{timestamp}/wo_pr",
        "w/o EL": f"../results/ablation/{timestamp}/wo_el"
    }

    results_summary = []

    for model_name, path in experiments.items():
        test_csvs = find_test_csvs_with_label(path)
        if not test_csvs:
            print(f"Warning: No test.csv found for {model_name} in {path}")
            continue

        # 按 label 分组聚合
        for label_name, csv_file in test_csvs:
            df = pd.read_csv(csv_file)
            
            # 计算均值和标准差
            metrics = {"RMSE": "repeat_rmse", "MAE": "repeat_mae", "$R^2$": "repeat_r2", "Pearson $r$": "pearson_corr"}
            
            res_row = {"Model": model_name, "Label": label_name}
            for disp_name, col_name in metrics.items():
                mean_val = df[col_name].mean()
                std_val = df[col_name].std()
                res_row[disp_name] = f"{mean_val:.4f} \pm {std_val:.4f}"
                
            results_summary.append(res_row)

    df_summary = pd.DataFrame(results_summary)
    
    # 1. 保存为 CSV
    csv_out = f"ablation_metrics_summary_{timestamp}.csv"
    df_summary.to_csv(csv_out, index=False)
    print(f"\n[Success] Ablation summary saved to {csv_out}\n")

    # 2. 生成 LaTeX 三线表 (使用 booktabs 规范)，按 Model 分组
    print("================== LaTeX Code for Paper ==================")
    
    # 获取所有唯一的 Model 和 Label
    models = df_summary["Model"].unique()
    labels = sorted(df_summary["Label"].unique())
    
    latex_str = "\\begin{table}[htbp]\n"
    latex_str += "\\centering\n"
    latex_str += "\\caption{Ablation Study Results on the Core Components}\n"
    
    # 动态生成表头：Model + 每个 Label 一列
    n_metrics = 4  # RMSE, MAE, R^2, Pearson r
    n_labels = len(labels)
    col_spec = "l" + "".join(["c" for _ in labels])  # Model 列 + 每组指标列
    latex_str += f"\\begin{{tabular}}{{{col_spec}}}\n"
    latex_str += "\\toprule\n"
    
    # 表头行1: Model名称 + 每个Label的列标题
    header_row1 = "Model"
    for label in labels:
        header_row1 += f" & \\multicolumn{{4}}{{c}}{{{label}}}"
    header_row1 += " \\\\"
    latex_str += header_row1 + "\n"
    
    # 表头行2: 各指标名称
    header_row2 = ""
    for i, label in enumerate(labels):
        if i == 0:
            header_row2 += " "  # Model列空白
        else:
            header_row2 += " & "
    header_row2 += "RMSE & MAE & $R^2$ & Pearson $r$"
    header_row2 = header_row2.strip() + " \\\\"
    latex_str += "\\midrule\n"
    
    # 数据行
    for model in models:
        row_data = [model]
        for label in labels:
            subset = df_summary[(df_summary["Model"] == model) & (df_summary["Label"] == label)]
            if not subset.empty:
                row_data.append(f"${subset['RMSE'].values[0]}$ & ${subset['MAE'].values[0]}$ & ${subset['$R^2$'].values[0]}$ & ${subset['Pearson $r$'].values[0]}$")
            else:
                row_data.append("- & - & - & -")
        latex_str += " & ".join(row_data) + " \\\\\n"
        
    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\label{tab:ablation}\n"
    latex_str += "\\end{table}"
    
    print(latex_str)
    print("==========================================================")

if __name__ == "__main__":
    main()
