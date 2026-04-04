import torch
import argparse
import os
import json
import numpy as np
from dataset import dataset
from run import evaluate
from torch_geometric.loader import DataLoader

def main():
    parser = argparse.ArgumentParser(description="Standalone Test Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_validation.pth")
    
    # 目标数据集参数
    parser.add_argument("--dataset_name", type=str, default="S1200")
    parser.add_argument("--label_type", type=str, required=True)
    parser.add_argument("--atlas_name", type=str, default="bna246")
    parser.add_argument("--sc_kinds", type=str, nargs="+", default=["FA", "fiber_count"])
    parser.add_argument("--fc_kind", type=str, default="pcc_rest")
    
    # 推理划分参数 (支持在不同数据集的不同划分下测试)
    parser.add_argument("--split_ratio", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition", type=str, default="test", choices=["train", "val", "test", "all"], 
                        help="Partition to evaluate on")
    
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--test_repeat", type=int, default=1)
    parser.add_argument("--no_normalize_labels", action="store_true")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟 run.py 的 args 对象
    class MockArgs:
        def __init__(self, repeat):
            self.test_repeat = repeat

    # ==========================================
    # 1. 恢复训练时的均值和标准差 (反标准化需要)
    # ==========================================
    meta_path = os.path.join(os.path.dirname(os.path.dirname(args.model_path)), "run_meta.json")
    if args.no_normalize_labels:
        lb_mean = torch.tensor(0.0, dtype=torch.float64, device=device)
        lb_std = torch.tensor(1.0, dtype=torch.float64, device=device)
        print(">>> Label normalization: DISABLED")
    else:
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Cannot find run_meta.json at {meta_path}. Required for label de-normalization.")
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        print(">>> Restoring Source Training Dataset to compute normalization params...")
        torch.manual_seed(meta.get("seed", 42))
        src_dt = dataset(
            dsType=meta.get("dataset", "HCD"),
            labelType=args.label_type,
            use_dataset_cfg=meta.get("use_dataset_cfg", True),
            dataset_name=meta.get("dataset_name", "S1200"),
            atlas_name=meta.get("atlas_name", "bna246"),
            sc_kinds=meta.get("sc_kinds", ["FA", "fiber_count"]),
            fc_kind=meta.get("fc_kind", "pcc_rest")
        )
        src_dt.setsubset(
            labelType=args.label_type, 
            labeldim=246, 
            split_ratio=meta.get("split_ratio", [0.7, 0.15, 0.15]), 
            create_val=True
        )
        train_labels = torch.tensor([data[1] for data in src_dt.train_dataset], dtype=torch.float64)
        lb_mean = train_labels.mean().to(device)
        lb_std = train_labels.std().to(device)
        print(f">>> Restored Source Normalization: mean={lb_mean.item():.4f}, std={lb_std.item():.4f}")

    # ==========================================
    # 2. 构建目标测试数据集
    # ==========================================
    print(f"\n>>> Building Target Dataset: {args.dataset_name}, Partition: {args.partition}")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)

    target_dt = dataset(
        dsType=args.dataset_name,
        labelType=args.label_type,
        use_dataset_cfg=True,
        dataset_name=args.dataset_name,
        atlas_name=args.atlas_name,
        sc_kinds=args.sc_kinds,
        fc_kind=args.fc_kind,
    )
    target_dt.setsubset(labelType=args.label_type, labeldim=246, split_ratio=args.split_ratio, create_val=True)
    
    if args.partition == "test":
        eval_loader = target_dt.test_dataloader()
    elif args.partition == "val":
        eval_loader = target_dt.val_dataloader()
    elif args.partition == "train":
        eval_loader = DataLoader(target_dt.train_dataset, batch_size=args.batch, shuffle=False)
    elif args.partition == "all":
        eval_loader = DataLoader(target_dt.datalist, batch_size=args.batch, shuffle=False)

    # ==========================================
    # 3. 推理（权重由 evaluate 从 --model_path 加载；输出写入独立子目录，避免覆盖训练时的 test.csv）
    # ==========================================
    output_dir = os.path.join(os.path.dirname(args.model_path), f"eval_{args.dataset_name}_{args.partition}")
    os.makedirs(output_dir, exist_ok=True)
    print(f">>> Output will be saved to: {output_dir}")

    evaluate(
        MockArgs(args.test_repeat),
        eval_loader,
        device,
        output_dir,
        lb_mean,
        lb_std,
        age_scale=100.0,
        checkpoint_path=args.model_path,
    )

if __name__ == "__main__":
    main()
