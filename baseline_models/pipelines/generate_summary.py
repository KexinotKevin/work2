#!/usr/bin/env python3
"""
根据过往实验记录重新生成汇总结果 CSV。

用法：
    python generate_summary.py --timestamp 20260405_144555 [--output_dir ./baseline_models/results/debug_baseline]

该脚本会扫描指定时间戳目录下的所有实验结果，
读取每个实验的 run_meta.json 和 test.csv，汇总生成 summary_results.csv。
"""

import argparse
import os
import sys
import glob
import json
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


def sanitize_name(text):
    """清理字符串，移除非法文件名字符。"""
    keep = []
    for ch in str(text):
        keep.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "unnamed"


def find_all_experiments(base_dir: str, timestamp: str) -> list:
    """扫描目录下所有包含 test.csv 的实验目录。
    
    目录结构：
        {base_dir}/{timestamp}/{timestamp}/
            ├── {ModelType}_{Modality}/
            │   └── {atlas}__sc_{sc_kinds}__fc_{fc_kind}/
            │       └── {split_tag}/
            │           └── seed_{seed}/
            │               ├── label_{label_name}/
            │               │   ├── test.csv
            │               │   └── ...
            │               └── run_meta.json
    
    返回：[(experiment_dir, meta_info, label_name), ...]
    """
    timestamp_dir = os.path.join(base_dir, timestamp, timestamp)
    if not os.path.exists(timestamp_dir):
        raise FileNotFoundError(f"时间戳目录不存在: {timestamp_dir}")
    
    experiments = []
    
    # 遍历所有 ModelType_Modality 组合
    for model_mod_dir in os.listdir(timestamp_dir):
        model_mod_path = os.path.join(timestamp_dir, model_mod_dir)
        if not os.path.isdir(model_mod_path):
            continue
        
        # 解析模型类型和模态 (格式: ModelType_Modality)
        if "_" not in model_mod_dir:
            print(f"警告：跳过无法解析的目录 {model_mod_dir}")
            continue
        
        # 尝试多种分隔方式：先按最后一个下划线分割模态
        parts = model_mod_dir.split("_")
        modality = parts[-1]  # SC, FC, SC_FC
        model_type = "_".join(parts[:-1])  # BNT, BrainRGIN, BrainNetworkTransformer 等
        
        # 遍历 atlas 配置目录
        for atlas_dir in os.listdir(model_mod_path):
            atlas_path = os.path.join(model_mod_path, atlas_dir)
            if not os.path.isdir(atlas_path):
                continue
            
            # 遍历 split 目录
            for split_dir in os.listdir(atlas_path):
                split_path = os.path.join(atlas_path, split_dir)
                if not os.path.isdir(split_path):
                    continue
                
                # 遍历 seed 目录
                for seed_dir in os.listdir(split_path):
                    seed_path = os.path.join(split_path, seed_dir)
                    if not os.path.isdir(seed_path):
                        continue
                    
                    # 读取 run_meta.json
                    meta_path = os.path.join(seed_path, "run_meta.json")
                    if not os.path.exists(meta_path):
                        print(f"警告：run_meta.json 不存在 {meta_path}")
                        continue
                    
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    
                    # 遍历 label 目录
                    for label_dir in os.listdir(seed_path):
                        label_path = os.path.join(seed_path, label_dir)
                        if not os.path.isdir(label_path) or not label_dir.startswith("label_"):
                            continue
                        
                        test_csv_path = os.path.join(label_path, "test.csv")
                        if not os.path.exists(test_csv_path):
                            continue
                        
                        label_name = label_dir.replace("label_", "")
                        experiments.append({
                            "dir": label_path,
                            "model_type": model_type,
                            "modality": modality,
                            "label_name": label_name,
                            "meta": meta,
                        })
    
    return experiments


def read_test_results(csv_path: str) -> dict:
    """读取单个 test.csv，计算汇总指标。
    
    test.csv 包含多行 repeat 结果（如 11 次重复）。
    返回：{metric_name: (mean, std), ...}
    其中 metric_name 会被规范化为大写格式（如 RMSE 而非 repeat_rmse）。
    """
    df = pd.read_csv(csv_path)
    
    # test.csv 列名到汇总列名的映射（统一为与 run_baseline.py 兼容的大写格式）
    column_rename = {
        "repeat_rmse": "RMSE",
        "repeat_mae": "MAE",
        "repeat_r2": "R2",
        "pearson_corr": "Pearson_r",
        "bias_age_corr": "Bias_Age_Corr",
    }
    
    results = {}
    for col in df.columns:
        values = df[col].dropna().values
        if len(values) > 0:
            # 映射到规范列名，不存在的列名直接使用原名（保持向后兼容）
            normalized = column_rename.get(col, col)
            results[normalized] = (np.mean(values), np.std(values))
    
    return results


def generate_summary(base_dir: str, timestamp: str, output_dir: str = None):
    """根据时间戳目录生成汇总结果。
    
    Args:
        base_dir: 基准目录 (如 ./baseline_models/results/debug_baseline)
        timestamp: 时间戳 (如 20260405_144555)
        output_dir: 可选，输出目录，默认覆盖原 summary_results.csv
    """
    print(f"\n{'='*60}")
    print(f"生成汇总结果: {timestamp}")
    print(f"基准目录: {base_dir}")
    print(f"{'='*60}\n")
    
    # 查找所有实验
    experiments = find_all_experiments(base_dir, timestamp)
    print(f"找到 {len(experiments)} 个实验结果\n")
    
    if not experiments:
        print("未找到任何实验结果！")
        return
    
    # 按 (model_type, modality, label_name) 分组
    summary_rows = []
    
    # 收集所有唯一的 (model, modality, label) 组合
    unique_combos = set()
    for exp in experiments:
        key = (exp["model_type"], exp["modality"])
        unique_combos.add(key)
    
    print(f"发现 {len(unique_combos)} 个模型-模态组合:")
    for combo in sorted(unique_combos):
        print(f"  - {combo[0]} + {combo[1]}")
    print()
    
    # 按模型-模态-标签逐个处理
    for model_type, modality in sorted(unique_combos):
        # 找到该组合下的所有实验
        combo_exps = [e for e in experiments 
                      if e["model_type"] == model_type and e["modality"] == modality]
        
        # 收集所有标签
        labels_in_combo = set(e["label_name"] for e in combo_exps)
        
        for label in sorted(labels_in_combo):
            label_exps = [e for e in combo_exps if e["label_name"] == label]
            
            # 获取 GRL 设置（使用第一个实验的设置）
            use_grl = label_exps[0]["meta"].get("use_grl", False)
            adv_weight = label_exps[0]["meta"].get("adv_weight", 0.001)
            
            # 读取所有重复实验的结果
            all_metrics = {}
            for exp in label_exps:
                test_csv = os.path.join(exp["dir"], "test.csv")
                try:
                    metrics = read_test_results(test_csv)
                    for metric_name, (mean, std) in metrics.items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append((mean, std))
                except Exception as e:
                    print(f"警告：读取 {test_csv} 失败: {e}")
            
            # 构建行数据
            row = {
                "Model": model_type,
                "Modality": modality,
                "Label": label,
                "Use_GRL": "Yes" if use_grl else "No",
                "Adv_Weight": adv_weight,
            }
            
            # 添加每个指标的汇总值（跨 seed 的平均）
            for metric_name, values in sorted(all_metrics.items()):
                # values: [(mean1, std1), (mean2, std2), ...] 每个 seed 的结果
                # 汇总：取所有 seed 均值的均值 和 标准差的均方根
                means = [v[0] for v in values]
                stds = [v[1] for v in values]
                
                final_mean = np.mean(means)
                final_std = np.sqrt(np.mean([s**2 for s in stds]))  # RMS of stds
                
                col_name = f"{label}_{metric_name}"
                row[col_name] = f"{final_mean:.4f}±{final_std:.4f}"
            
            summary_rows.append(row)
            print(f"  处理: {model_type} + {modality} + {label}")
    
    # 生成 DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    # 保存结果
    timestamp_dir = os.path.join(base_dir, timestamp, timestamp)
    if output_dir:
        timestamp_dir = output_dir
    
    summary_path = os.path.join(timestamp_dir, "summary_results.csv")
    os.makedirs(timestamp_dir, exist_ok=True)
    
    # 如果文件已存在，读取并追加（覆盖重复的，添加新的）
    if os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        # 按 (Model, Modality, Label) 去重，保留新结果
        key_cols = ["Model", "Modality", "Label"]
        combined = pd.concat([existing_df, summary_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep='last')
        summary_df = combined
    
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"汇总结果已保存到: {summary_path}")
    print(f"共 {len(summary_df)} 行结果")
    print(f"{'='*60}")
    
    return summary_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据过往实验记录重新生成汇总结果 CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
    # 重新生成指定时间戳的汇总结果
    python generate_summary.py --timestamp 20260405_144555
    
    # 指定不同的基准目录
    python generate_summary.py --timestamp 20260405_144555 --base_dir ./baseline_models/results/debug_baseline
    
    # 输出到指定目录（不覆盖原文件）
    python generate_summary.py --timestamp 20260405_144555 --output_dir /tmp/summary
        """
    )
    
    parser.add_argument(
        "--timestamp", "-t",
        type=str,
        required=True,
        help="实验时间戳 (如 20260405_144555)"
    )
    
    parser.add_argument(
        "--base_dir", "-b",
        type=str,
        default="./baseline_models/results/debug_baseline",
        help="基准目录路径 (默认: ./baseline_models/results/debug_baseline)"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=None,
        help="输出目录路径 (默认覆盖原 summary_results.csv)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 转换为绝对路径
    base_dir = os.path.abspath(args.base_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else None
    
    generate_summary(
        base_dir=base_dir,
        timestamp=args.timestamp,
        output_dir=output_dir
    )
