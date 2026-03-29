import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vis import build_saliency_matrix, plot_saliency_results, get_coords  # 新增
from nilearn.datasets import fetch_atlas_aal  # 新增
import nibabel as nib  # 新增


def read_label_name_from_dir(label_dir):
    label_name_file = os.path.join(label_dir, "label_name.txt")
    if os.path.isfile(label_name_file):
        with open(label_name_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    base = os.path.basename(label_dir)
    return base[len("label_") :] if base.startswith("label_") else base


def collect_label_results(combo_dir):
    rows = []
    label_entries = []
    for name in sorted(os.listdir(combo_dir)):
        label_dir = os.path.join(combo_dir, name)
        if not os.path.isdir(label_dir) or not name.startswith("label_"):
            continue

        display_label = read_label_name_from_dir(label_dir)
        label_entries.append((label_dir, display_label))

        loss_csv = os.path.join(label_dir, "loss.csv")
        test_csv = os.path.join(label_dir, "test.csv")
        if not os.path.isfile(loss_csv) or not os.path.isfile(test_csv):
            continue

        loss_df = pd.read_csv(loss_csv)
        test_df = pd.read_csv(test_csv)
        rows.append(
            {
                "label": display_label,
                "best_train_loss": float(loss_df["train_loss"].min()),
                "best_val_loss": float(loss_df["val_loss"].min()),
                "final_train_loss": float(loss_df["train_loss"].iloc[-1]),
                "final_val_loss": float(loss_df["val_loss"].iloc[-1]),
                "mean_rmse": float(test_df["repeat_rmse"].mean()),
                "std_rmse": float(test_df["repeat_rmse"].std(ddof=0)),
                "mean_r2": float(test_df["repeat_r2"].mean()),
                "std_r2": float(test_df["repeat_r2"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows), label_entries


def plot_loss_curves(label_entries, out_dir):
    plt.figure(figsize=(8, 5))
    for label_dir, label_name in label_entries:
        loss_csv = os.path.join(label_dir, "loss.csv")
        if not os.path.isfile(loss_csv):
            continue
        loss_df = pd.read_csv(loss_csv)
        plt.plot(loss_df["val_loss"].values, label=label_name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE")
    plt.title("Validation Curves Across Labels")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_loss_curves.png"), dpi=300)
    plt.close()


def plot_metric_bars(summary_df, out_dir):
    if summary_df.empty:
        return

    x = np.arange(len(summary_df))
    labels = summary_df["label"].tolist()

    plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
    plt.bar(x, summary_df["mean_rmse"].values, yerr=summary_df["std_rmse"].values, capsize=4)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("RMSE")
    plt.title("Test RMSE (mean +/- std)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "test_rmse_bar.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
    plt.bar(x, summary_df["mean_r2"].values, yerr=summary_df["std_r2"].values, capsize=4)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("R2")
    plt.title("Test R2 (mean +/- std)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "test_r2_bar.png"), dpi=300)
    plt.close()


def plot_interpretability(combo_dir, out_dir, coords):
    """自动遍历各个预测任务，生成 Saliency 解释图"""
    for name in sorted(os.listdir(combo_dir)):
        label_dir = os.path.join(combo_dir, name)
        if not os.path.isdir(label_dir) or not name.startswith("label_"):
            continue
        
        display_label = read_label_name_from_dir(label_dir)
        sal_file = os.path.join(label_dir, "saliency_maps.npy")
        idx_file = os.path.join(label_dir, "edge_indices.npy")
        
        if os.path.isfile(sal_file) and os.path.isfile(idx_file):
            print(f"Plotting Saliency Maps for {display_label}...")
            saliency_data = np.load(sal_file, allow_pickle=True)
            edge_indices = np.load(idx_file, allow_pickle=True)
            
            # 将每个样本转为动态大小的邻接矩阵
            sal_matrices = []
            for e_idx, s_attr in zip(edge_indices, saliency_data):
                sal_matrices.append(build_saliency_matrix(e_idx, s_attr))
            
            # 调用 vis.py 进行画图
            plot_saliency_results(sal_matrices, coords, out_dir, display_label)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot result graphs for multi-label runs")
    parser.add_argument("--combo_dir", type=str, required=True)
    parser.add_argument("--plots_dir", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    combo_dir = args.combo_dir
    if not os.path.isdir(combo_dir):
        raise FileNotFoundError(f"combo_dir not found: {combo_dir}")

    out_dir = args.plots_dir if args.plots_dir else os.path.join(combo_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    summary_df, label_entries = collect_label_results(combo_dir)
    if summary_df.empty:
        raise RuntimeError(f"No valid label results found under {combo_dir}")

    summary_df.to_csv(os.path.join(out_dir, "summary_by_label.csv"), index=False)
    plot_loss_curves(label_entries, out_dir)
    plot_metric_bars(summary_df, out_dir)

    # ------------------ 新增调用部分 ------------------
    print("Loading Brain Atlas coordinates...")
    atlas = fetch_atlas_aal()
    # 注意: fetch_atlas_aal 提供 116 节点坐标。如果在其它数据集(如246节点)上运行，
    # vis.py 中的安全检查会自动跳过 3D 脑图，但热力图依然会成功生成。
    coords = get_coords(nib.load(atlas['maps'])) 
    
    plot_interpretability(combo_dir, out_dir, coords)
    # --------------------------------------------------

    meta_path = os.path.join(combo_dir, "run_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(os.path.join(out_dir, "run_meta_snapshot.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=True)

    print(f"saved plots and summary to: {out_dir}")


if __name__ == "__main__":
    main()
