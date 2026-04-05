import os
import sys
import glob
import pandas as pd
import numpy as np

def find_latest_test_csv(base_dir):
    """在指定的消融变体目录下，找到最新的 test.csv"""
    search_pattern = os.path.join(base_dir, '**', 'test.csv')
    files = glob.glob(search_pattern, recursive=True)
    if not files:
        return None
    # 按修改时间排序，取最新的
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def main():
    # 定义四个消融实验及其对应的文件夹
    timestamp = sys.argv[1]
    experiments = {
        "Ours": f"./results/{timestamp}/ours",
        "w/o GRL": f"./results/{timestamp}/wo_grl",
        "w/o PR": f"./results/{timestamp}/wo_pr",
        "w/o EL": f"./results/{timestamp}/wo_el"
    }

    results_summary = []

    for model_name, path in experiments.items():
        csv_file = find_latest_test_csv(path)
        if csv_file is None:
            print(f"Warning: No test.csv found for {model_name} in {path}")
            continue

        df = pd.read_csv(csv_file)
        
        # 计算均值和标准差
        res_row = {"Model": model_name}
        metrics = {"RMSE": "repeat_rmse", "MAE": "repeat_mae", "$R^2$": "repeat_r2", "Pearson $r$": "pearson_corr"}
        
        for disp_name, col_name in metrics.items():
            mean_val = df[col_name].mean()
            std_val = df[col_name].std()
            res_row[disp_name] = f"{mean_val:.4f} \pm {std_val:.4f}"
            
        results_summary.append(res_row)

    df_summary = pd.DataFrame(results_summary)
    
    # 1. 保存为 CSV
    csv_out = "ablation_metrics_summary.csv"
    df_summary.to_csv(csv_out, index=False)
    print(f"\n[Success] Ablation summary saved to {csv_out}\n")

    # 2. 生成 LaTeX 三线表 (使用 booktabs 规范)
    print("================== LaTeX Code for Paper ==================")
    latex_str = "\\begin{table}[htbp]\n"
    latex_str += "\\centering\n"
    latex_str += "\\caption{Ablation Study Results on the Core Components}\n"
    latex_str += "\\begin{tabular}{lcccc}\n"
    latex_str += "\\toprule\n"
    latex_str += "Model & RMSE & MAE & $R^2$ & Pearson $r$ \\\\\n"
    latex_str += "\\midrule\n"
    
    for _, row in df_summary.iterrows():
        latex_str += f"{row['Model']} & ${row['RMSE']}$ & ${row['MAE']}$ & ${row['$R^2$']}$ & ${row['Pearson $r$']}$ \\\\\n"
        
    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\label{tab:ablation}\n"
    latex_str += "\\end{table}"
    
    print(latex_str)
    print("==========================================================")

if __name__ == "__main__":
    main()
