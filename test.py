import torch
import argparse
import os
import numpy as np
from dataset import dataset
from run import evaluate


def main():
    parser = argparse.ArgumentParser(description="Standalone Test Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_validation.pth")
    parser.add_argument("--dataset_name", type=str, default="S1200")
    parser.add_argument("--label_type", type=str, required=True)
    parser.add_argument("--atlas_name", type=str, default="bna246")
    parser.add_argument("--sc_kinds", type=str, nargs="+", default=["FA", "fiber_count"])
    parser.add_argument("--fc_kind", type=str, default="pcc_rest")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--test_repeat", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟 run.py 的 args 对象以复用 evaluate 函数
    class MockArgs:
        def __init__(self, repeat):
            self.test_repeat = repeat

    # 加载数据（使用 dataset_cfg 配置，与训练保持一致）
    dt = dataset(
        dsType="HCD",
        labelType=args.label_type,
        use_dataset_cfg=True,
        dataset_name=args.dataset_name,
        atlas_name=args.atlas_name,
        sc_kinds=args.sc_kinds,
        fc_kind=args.fc_kind,
    )
    # 确保划分 seed 与训练一致
    dt.setsubset(labelType=args.label_type, labeldim=246, split_ratio=[0.7, 0.15, 0.15], create_val=True)
    testloader = dt.test_dataloader()

    # 加载模型
    model = torch.load(args.model_path, weights_only=False).to(device)
    model.eval()

    # 执行测试（输出文件夹设为模型所在目录）
    output_dir = os.path.dirname(args.model_path)
    evaluate(MockArgs(args.test_repeat), testloader, device, output_dir)


if __name__ == "__main__":
    main()
