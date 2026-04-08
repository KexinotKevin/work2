import os
import sys
import glob
import pandas as pd
import numpy as np
import argparse

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

def find_test_csvs_with_label(base_dir):
    """在指定的消融变体目录下，找到所有 label_* 子目录下的 test.csv，返回 (label_name, csv_path) 列表
    
    支持两种路径结构：
    1. 旧结构: base_dir/{timestamp}/{variant}/label_*/test.csv
    2. 新结构: base_dir/{timestamp}/{variant}/{timestamp2}/HCD/{atlas}/split_*/seed_*/label_*/test.csv
    """
    results = []
    if not os.path.exists(base_dir):
        return results
    
    # 先尝试旧结构：直接在 base_dir 下找 label_* 目录
    csv_files = glob.glob(os.path.join(base_dir, 'label_*', 'test.csv'))
    
    # 如果旧结构没找到，尝试新结构：使用 glob 递归查找
    if not csv_files:
        csv_files = glob.glob(os.path.join(base_dir, '**', 'label_*', 'test.csv'), recursive=True)
    
    for csv_path in sorted(csv_files):
        label_dir = os.path.dirname(csv_path)
        label_name = os.path.basename(label_dir).replace("label_", "")
        results.append((label_name, csv_path))
    
    return results

def format_metric(value):
    """格式化单个指标值"""
    if pd.isna(value):
        return "-"
    return f"{value:.4f}"

def generate_csv(df_summary, labels, models, timestamp):
    """
    生成与 ablation_metrics_modules.csv 风格一致的 CSV 文件。
    格式：
    - 第1行: 4个标签名，每个标签4个指标
    - 第2行: Model, RMSE, MAE, $R^2$, Pearson $r$, ...
    - 第3行起: 数据行
    """
    csv_out = f"ablation/ablation_tables/ablation_metrics_summary_{timestamp}.csv"
    
    # 构建第1行：标签名 + 空值填充
    n_metrics = 5  # RMSE, MAE, R^2, Pearson r, CCC
    row1_cols = [""]
    for label in labels:
        row1_cols.extend([label, "", "", "", ""])  # 每个标签5个指标位
    
    # 构建第2行：Model, RMSE, MAE, $R^2$, Pearson $r$, CCC
    row2_cols = ["Model"]
    for _ in labels:
        row2_cols.extend(["RMSE", "MAE", "$R^2$", "Pearson $r$", "CCC"])
    
    # 构建数据行
    data_rows = []
    for model in models:
        row_data = [model]
        for label in labels:
            subset = df_summary[(df_summary["Model"] == model) & (df_summary["Label"] == label)]
            if not subset.empty:
                row_data.append(format_metric(subset["RMSE"].values[0]))
                row_data.append(format_metric(subset["MAE"].values[0]))
                row_data.append(format_metric(subset["R2"].values[0]))
                row_data.append(format_metric(subset["Pearson_r"].values[0]))
                row_data.append(format_metric(subset["CCC"].values[0]))
            else:
                row_data.extend(["-", "-", "-", "-", "-"])
        data_rows.append(row_data)
    
    # 写入CSV
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        writer = pd.DataFrame([row1_cols, row2_cols] + data_rows)
        writer.to_csv(f, header=False, index=False)
    
    return csv_out

def generate_latex(df_summary, labels, models, timestamp):
    """
    生成与 ablation_metrics_modules.csv 风格一致的 LaTeX 三线表。
    """
    csv_out = f"ablation/ablation_tables/ablation_metrics_summary_{timestamp}.tex"
    
    # 动态计算列数: 1 (Model) + len(labels) * 5 (每个label 5个指标)
    n_metrics = 5  # RMSE, MAE, R^2, Pearson r, CCC
    col_spec = "l" + "".join(["c" for _ in range(len(labels) * n_metrics)])
    
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Ablation Study Results on the Core Components}")
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    # 表头行1: Model + 每个Label的列标题
    header_row1 = "\\textbf{Model}"
    for label in labels:
        # 简化标签名用于显示
        label_display = label.replace("nih_", "").replace("_unadjusted", "")
        header_row1 += f" & \\multicolumn{{4}}{{c}}{{\\textbf{{{label_display}}}}}"
    header_row1 += " \\\\"
    latex_lines.append(header_row1)
    
    # 表头行2: 各指标名称
    header_row2 = ""
    for i, label in enumerate(labels):
        if i == 0:
            header_row2 += "\\textbf{}"  # Model列空白
        else:
            header_row2 += " & \\textbf{}"
    header_row2 += " & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{R$^2$} & \\textbf{Pearson $r$} & \\textbf{CCC}"
    header_row2 = header_row2.strip() + " \\\\"
    latex_lines.append("\\midrule")
    latex_lines.append(header_row2)
    
    # 数据行
    for model in models:
        row_data = [model]
        for label in labels:
            subset = df_summary[(df_summary["Model"] == model) & (df_summary["Label"] == label)]
            if not subset.empty:
                row_data.append(format_metric(subset["RMSE"].values[0]))
                row_data.append(format_metric(subset["MAE"].values[0]))
                row_data.append(format_metric(subset["R2"].values[0]))
                row_data.append(format_metric(subset["Pearson_r"].values[0]))
                row_data.append(format_metric(subset["CCC"].values[0]))
            else:
                row_data.extend(["-", "-", "-", "-", "-"])
        latex_lines.append(" & ".join(row_data) + " \\\\")
        
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\label{tab:ablation}")
    latex_lines.append("\\end{table}")
    
    latex_content = "\n".join(latex_lines)
    
    with open(csv_out, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    return csv_out, latex_content

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成消融实验汇总表格")
    parser.add_argument("--ablation_exp", type=str, required=True,
                        help="消融实验根目录路径 (包含 ours, wo_grl, wo_pr, wo_el 子目录)")
    parser.add_argument("--ours_exp", type=str, required=True,
                        help=" Ours 实验数据目录路径 (包含 label_* 子目录)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 定义消融实验及其对应的文件夹 (相对于 ablation_exp)
    ablation_exp_path = args.ablation_exp
    ours_exp_path = args.ours_exp
    
    experiments = {
        "Ours": ours_exp_path,
        "w/o GRL": os.path.join(ablation_exp_path, "wo_grl"),
        "w/o PR": os.path.join(ablation_exp_path, "wo_pr"),
        "w/o EL": os.path.join(ablation_exp_path, "wo_el")
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
            
            # 计算均值
            res_row = {
                "Model": model_name, 
                "Label": label_name,
                "RMSE": df["repeat_rmse"].mean(),
                "MAE": df["repeat_mae"].mean(),
                "R2": df["repeat_r2"].mean(),
                "Pearson_r": df["pearson_corr"].mean(),
                "CCC": df["repeat_ccc"].mean()
            }
            results_summary.append(res_row)

    if not results_summary:
        print("No results found.")
        return

    df_summary = pd.DataFrame(results_summary)
    
    # 获取所有唯一的 Model 和 Label
    models = df_summary["Model"].unique()
    labels = sorted(df_summary["Label"].unique())
    
    # 生成时间戳用于文件名
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存为 CSV
    csv_out = generate_csv(df_summary, labels, models, timestamp)
    print(f"\n[Success] Ablation summary saved to {csv_out}")
    
    # 2. 生成并保存 LaTeX 表格
    tex_out, latex_content = generate_latex(df_summary, labels, models, timestamp)
    print(f"[Success] LaTeX table saved to {tex_out}")
    
    print("\n================== LaTeX Code for Paper ==================")
    print(latex_content)
    print("==========================================================")

if __name__ == "__main__":
    main()
