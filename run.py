import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset import dataset
from model_r import LGUNet_rela


class DynamicLearningRateScheduler:
    """动态学习率调度器，结合loss plateau检测和梯度监控
    
    策略：
    1. 当loss plateau时降低学习率（patience轮内未改善）
    2. 当梯度范数持续过小时适当增大学习率
    3. 当梯度范数过大时减小学习率防止震荡
    4. 支持warmup阶段平滑起步
    """
    
    def __init__(self, optimizer, base_lr, min_lr=1e-6, 
                 patience=10, factor=0.5, warmup_epochs=5,
                 grad_norm_thresh_low=0.01, grad_norm_thresh_high=10.0,
                 increase_lr_factor=1.05, window_size=5):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor
        self.warmup_epochs = warmup_epochs
        self.grad_norm_thresh_low = grad_norm_thresh_low
        self.grad_norm_thresh_high = grad_norm_thresh_high
        self.increase_lr_factor = increase_lr_factor
        self.window_size = window_size
        
        self.current_lr = base_lr
        self.best_loss = float('inf')
        self.wait_count = 0
        self.grad_history = []
        self.loss_history = []
        self.increase_count = 0  # 防止无限增大学习率
        
    def step(self, epoch, loss, grad_norm=None):
        """更新学习率
        
        Args:
            epoch: 当前epoch
            loss: 当前epoch的loss
            grad_norm: 当前batch的梯度范数（可选）
        """
        self.loss_history.append(loss)
        
        # Warmup阶段：线性增加学习率
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            self._set_lr(lr)
            return lr
            
        # 记录梯度范数
        if grad_norm is not None:
            self.grad_history.append(grad_norm)
            if len(self.grad_history) > self.window_size:
                self.grad_history.pop(0)
        
        # 策略1: Loss plateau检测
        if loss < self.best_loss - 1e-6:
            self.best_loss = loss
            self.wait_count = 0
        else:
            self.wait_count += 1
            
        # 策略2: 基于梯度范数的调整
        lr_adjustment = 1.0
        if len(self.grad_history) >= self.window_size:
            avg_grad = np.mean(self.grad_history)
            
            # 梯度过小：可能陷入plateau，适当增大学习率
            if avg_grad < self.grad_norm_thresh_low and self.increase_count < 3:
                lr_adjustment = self.increase_lr_factor
                self.increase_count += 1
            # 梯度过大：可能震荡，减小学习率
            elif avg_grad > self.grad_norm_thresh_high:
                lr_adjustment = 0.7
                self.increase_count = 0
            else:
                self.increase_count = 0
        
        # 综合调整
        if self.wait_count >= self.patience:
            new_lr = max(self.current_lr * self.factor, self.min_lr)
            self.wait_count = 0
            self.increase_count = 0  # 重置增加计数
        else:
            new_lr = self.current_lr * lr_adjustment
            new_lr = max(new_lr, self.min_lr)
            new_lr = min(new_lr, self.base_lr)  # 不超过初始学习率
            
        if abs(new_lr - self.current_lr) > 1e-9:
            self._set_lr(new_lr)
            
        return self.current_lr
    
    def _set_lr(self, lr):
        """设置所有参数组的学习率"""
        self.current_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_lr(self):
        return self.current_lr


def setup_logging(output_root, timestamp=None):
    """设置日志文件，记录终端输出
    
    Args:
        output_root: 输出根目录
        timestamp: 如果提供，则使用该时间戳创建日志；否则生成新的
    Returns:
        timestamp: 使用的时间戳
        log_file: 日志文件路径
    """
    if timestamp is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_dir = os.path.join(output_root, "train_log")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "{}.log".format(timestamp))
    
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding="utf-8")
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout
    return timestamp, log_file


def remove_pattern(pattern):
    for f in glob.glob(pattern):
        if os.path.isfile(f):
            os.remove(f)


def sanitize_name(text):
    keep = []
    for ch in str(text):
        keep.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "unnamed"


def split_tag(split_ratio):
    vals = [int(round(float(x) * 100)) for x in split_ratio]
    return "split_{}_{}_{}".format(vals[0], vals[1], vals[2])


def build_combo_dir(args, timestamp):
    sc_tag = "-".join([sanitize_name(x) for x in args.sc_kinds_resolved])
    fc_tag = sanitize_name(args.fc_kind)
    combo_name = "atlas_{}__sc_{}__fc_{}".format(sanitize_name(args.atlas_name), sc_tag, fc_tag)
    return os.path.join(
        args.output_root,
        timestamp,
        sanitize_name(args.dataset_name if args.use_dataset_cfg else args.dataset),
        combo_name,
        split_tag(args.split_ratio),
        "seed_{}".format(args.seed),
    )


# 注意：函數簽名增加了 lb_mean, lb_std, age_scale
def train(args, model, trainloader, valloader, optimizer, scheduler, device, label_output_dir, lb_mean, lb_std, age_scale=100.0, use_dynamic_lr=False, dynamic_scheduler=None):
    train_loss = []
    val_loss = []
    lr_history = []
    grad_norm_history = []
    tmp_train = os.path.join(label_output_dir, "bt_tmp.pth")
    tmp_val = os.path.join(label_output_dir, "bv_tmp.pth")
    remove_pattern(os.path.join(label_output_dir, "bt_tmp*.pth"))
    remove_pattern(os.path.join(label_output_dir, "bv_tmp*.pth"))

    for epoch in range(args.num_epochs):
        t = time.time()
        train_loss_tmp = []
        grad_norms = []
        model.train()
        for g_data, lb_data in trainloader:
            g_data = g_data.to(device)
            lb_data = lb_data.to(device)
            optimizer.zero_grad()

            # 前向傳播：主任務 + GRL 對抗任務
            out_cog, out_age, out_gender = model(g_data, lb_data, g_data.batch)

            # ====== 【核心修改：標籤標準化】 ======
            lb_data_norm = (lb_data - lb_mean) / (lb_std + 1e-8)
            
            # 主任務損失：讓模型去擬合標準化後的標籤
            loss_cog = F.smooth_l1_loss(out_cog.squeeze(-1), lb_data_norm)

            # ====== 【修復 1：歸一化年齡標籤，防止 GRL 梯度爆炸】 ======
            # 將數值除以 age_scale 壓縮到 0~1 附近，使其梯度量級與 loss_cog 對齊
            age_labels = g_data.age.squeeze(-1).to(device) / age_scale
            out_age_scaled = out_age.squeeze(-1) / age_scale
            loss_age = F.smooth_l1_loss(out_age_scaled, age_labels)
            
            loss_gender = F.binary_cross_entropy_with_logits(out_gender.squeeze(-1), g_data.gender.squeeze(-1).to(device))

            # ====== 【修復 2：增加對抗任務權重，防止喧賓奪主】 ======
            # GRL 是一種強正則化，權重 (adv_weight) 通常設為 0.01 ~ 0.1 之間
            adv_weight = 0.05 
            loss = loss_cog + adv_weight * (loss_age + loss_gender)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            # 记录梯度范数（用于动态学习率调整）
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            grad_norms.append(total_grad_norm)
            
            optimizer.step()
            
            # ====== 【修復 3：統一量綱】 ======
            # 訓練日誌只記錄主任務 loss_cog，使其與 loss_val 完全具備可比性
            train_loss_tmp.append(loss_cog.item())
        
        # 计算平均梯度范数和训练loss
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        grad_norm_history.append(avg_grad_norm)
        train_loss_v = sum(train_loss_tmp) / len(train_loss_tmp)
        
        # 更新学习率
        if use_dynamic_lr and dynamic_scheduler is not None:
            current_lr = dynamic_scheduler.step(epoch, train_loss_v, avg_grad_norm)
        else:
            if scheduler is not None:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
        lr_history.append(current_lr)

        model.eval()
        val_loss_tmp = []
        with torch.no_grad():
            for g_data, lb_data in valloader:
                g_data = g_data.to(device)
                lb_data = lb_data.to(device)
                out_cog, _, _ = model(g_data, lb_data, g_data.batch)
                
                # ====== 【核心修改：驗證集也需要計算標準化後的 Loss】 ======
                lb_data_norm = (lb_data - lb_mean) / (lb_std + 1e-8)
                loss = F.smooth_l1_loss(out_cog.squeeze(-1), lb_data_norm)
                
                val_loss_tmp.append(loss.item())
        val_loss_v = sum(val_loss_tmp) / len(val_loss_tmp)

        train_loss.append(train_loss_v)
        val_loss.append(val_loss_v)
        print(
            "Epoch: {:04d}".format(epoch + 1),
            "loss_train: {:.4f}".format(train_loss_v),
            "loss_val: {:.4f}".format(val_loss_v),
            "lr: {:.6f}".format(current_lr),
            "grad_norm: {:.4f}".format(avg_grad_norm),
            "time: {:.4f}s".format(time.time() - t),
        )

        if train_loss_v <= min(train_loss):
            torch.save(model, tmp_train)
        if val_loss_v <= min(val_loss):
            torch.save(model, tmp_val)

    best_train_path = os.path.join(label_output_dir, "best_train.pth")
    best_val_path = os.path.join(label_output_dir, "best_validation.pth")
    if os.path.exists(best_train_path):
        os.remove(best_train_path)
    if os.path.exists(best_val_path):
        os.remove(best_val_path)
    os.replace(tmp_train, best_train_path)
    os.replace(tmp_val, best_val_path)

    df = pd.DataFrame(columns=["train_loss", "val_loss", "learning_rate", "grad_norm"])
    df["train_loss"] = train_loss
    df["val_loss"] = val_loss
    df["learning_rate"] = lr_history
    df["grad_norm"] = grad_norm_history
    df.to_csv(os.path.join(label_output_dir, "loss.csv"), index=False)


# 注意：函數簽名增加了 age_scale
def evaluate(args, testloader, device, label_output_dir, lb_mean, lb_std, age_scale=100.0):
    model_t = torch.load(
        os.path.join(label_output_dir, "best_validation.pth"),
        weights_only=False,
    ).to(device)
    model_t.eval()

    from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr

    total_rmse, total_mae, total_r2, total_p = [], [], [], []
    
    print(f"\n>>> Starting Evaluation for Label: {label_output_dir}")
    for _ in range(args.test_repeat):
        with torch.no_grad():
            lb_p = []
            lb_t = []
            for g_test, lb_test in testloader:
                g_test = g_test.to(device)
                lb_test = lb_test.to(device)
                
                # 評估時僅使用主任務輸出 (此時預測的是標準化後的值)
                lb_pred, _, _ = model_t(g_test, lb_test, g_test.batch)
                
                # ====== 【核心修改：反標準化，還原為真實量綱】 ======
                lb_pred_real = lb_pred.squeeze(-1) * lb_std + lb_mean
                
                lb_t.append(lb_test.cpu().numpy())
                lb_p.append(lb_pred_real.cpu().numpy()) # 記錄還原後的值
            
            lb_test = np.array(lb_t).flatten()
            lb_pred = np.array(lb_p).flatten()
            
            total_rmse.append(root_mean_squared_error(lb_test, lb_pred))
            total_mae.append(mean_absolute_error(lb_test, lb_pred))
            total_r2.append(r2_score(lb_test, lb_pred))
            total_p.append(pearsonr(lb_test, lb_pred)[0])

    # 打印到終端
    print(f"Test Results over {args.test_repeat} repeats:")
    print(f"  RMSE: {np.mean(total_rmse):.4f} ± {np.std(total_rmse):.4f}")
    print(f"  MAE:  {np.mean(total_mae):.4f} ± {np.std(total_mae):.4f}")
    print(f"  R2:   {np.mean(total_r2):.4f} ± {np.std(total_r2):.4f}")
    print(f"  Pearson r: {np.mean(total_p):.4f} ± {np.std(total_p):.4f}")

    # 保存到 CSV
    df2 = pd.DataFrame({
        "repeat_rmse": total_rmse,
        "repeat_mae": total_mae,
        "repeat_r2": total_r2,
        "pearson_corr": total_p
    })
    df2.to_csv(os.path.join(label_output_dir, "test.csv"), index=False)


# 注意：函數簽名增加了 age_scale
def explain_model(args, testloader, device, label_output_dir, lb_mean, lb_std, age_scale=100.0):
    print("\n>>> Extracting Edge-Major Saliency Maps...")
    model_t = torch.load(
        os.path.join(label_output_dir, "best_validation.pth"),
        weights_only=False,
    ).to(device)
    model_t.eval()

    from torch_geometric.loader import DataLoader
    expl_loader = DataLoader(testloader.dataset, batch_size=1, shuffle=False)

    all_sal_matrices = []

    for g_test, lb_test in expl_loader:
        g_test, lb_test = g_test.to(device), lb_test.to(device)
        g_test.edge_attr.requires_grad = True

        lb_pred, _, _ = model_t(g_test, lb_test, g_test.batch)
        
        # ====== 【核心修改：反標準化後再算解釋性Loss，確保梯度量級正確】 ======
        lb_pred_real = lb_pred.squeeze(-1) * lb_std + lb_mean
        loss = torch.abs(lb_pred_real - lb_test).mean()
        loss.backward()

        saliency = (g_test.edge_attr.grad * g_test.edge_attr).abs().detach().cpu().numpy()
        e_idx = g_test.edge_index.detach().cpu().numpy()
        saliency = np.nan_to_num(saliency, nan=0.0, posinf=0.0, neginf=0.0)

        num_nodes = int(np.max(e_idx)) + 1
        num_relations = saliency.shape[1] if len(saliency.shape) > 1 else 1
        
        sal_mat = np.zeros((num_nodes, num_nodes, num_relations))
        
        for r in range(num_relations):
            sal_r = saliency[:, r] if num_relations > 1 else saliency
            max_val, min_val = sal_r.max(), sal_r.min()
            if max_val > min_val:
                sal_r_norm = (sal_r - min_val) / (max_val - min_val)
            else:
                sal_r_norm = np.zeros_like(sal_r)
            
            for k in range(e_idx.shape[1]):
                i, j = int(e_idx[0, k]), int(e_idx[1, k])
                sal_mat[i, j, r] = sal_r_norm[k]
                sal_mat[j, i, r] = sal_r_norm[k]
                
        all_sal_matrices.append(sal_mat)

    np.save(os.path.join(label_output_dir, "saliency_matrices.npy"), np.array(all_sal_matrices))
    print(f"Interpretability evidence saved to {label_output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="LG-BrainUNet run entry")
    parser.add_argument("--dataset_class", type=str, default="dataset", choices=["dataset"])
    parser.add_argument("--dataset", type=str, default="HCD", choices=["HCP", "HCD"])
    parser.add_argument("--label_type", type=str, default="CogFluidComp_Unadj")
    parser.add_argument("--label_types", type=str, default="")

    parser.add_argument("--use_dataset_cfg", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="HCD")
    parser.add_argument("--atlas_name", type=str, default="bna246")
    parser.add_argument("--sc_kind", type=str, default="FA,fiber_count")
    parser.add_argument("--sc_kinds", type=str, nargs="+", default=None)
    parser.add_argument("--fc_kind", type=str, default="pcc_rest")

    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--l2_penalty", type=float, default=0.001)
    parser.add_argument("--input_dimension", type=int, default=246)
    parser.add_argument("--hidden_dimension", type=int, default=246)
    parser.add_argument("--output_dimension", type=int, default=1)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--pool_ratio", type=float, nargs="+", default=[0.5, 0.8, 0.5])

    parser.add_argument("--split_ratio", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_repeat", type=int, default=11)
    parser.add_argument("--output_root", type=str, default="./results")
    parser.add_argument("--normalize_labels", action="store_true", default=True,
                        help="Enable zscore normalization for cognition labels and age (default: enabled)")
    parser.add_argument("--no_normalize_labels", action="store_true",
                        help="Disable zscore normalization for cognition labels and age")
    
    # 动态学习率参数
    parser.add_argument("--use_dynamic_lr", action="store_true", default=False,
                        help="Enable dynamic learning rate adjustment based on loss and gradients")
    parser.add_argument("--lr_patience", type=int, default=10,
                        help="Patience for loss plateau detection in dynamic LR scheduler")
    parser.add_argument("--lr_factor", type=float, default=0.5,
                        help="Learning rate decay factor when plateau detected")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum learning rate for dynamic scheduler")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of warmup epochs for learning rate")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    # 先生成 timestamp，确保所有路径使用同一时间戳
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # 初始化日志，使用已有的 timestamp，并使用追加模式
    _, log_file = setup_logging(args.output_root, timestamp)
    args.sc_kinds_resolved = args.sc_kinds if args.sc_kinds is not None else [x.strip() for x in str(args.sc_kind).split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    labels = [x.strip() for x in str(args.label_types).split(",") if x.strip()]
    if not labels:
        labels = [args.label_type]

    combo_dir = build_combo_dir(args, timestamp)
    os.makedirs(combo_dir, exist_ok=True)
    print("result combo_dir: {}".format(combo_dir))
    run_meta = {
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "use_dataset_cfg": bool(args.use_dataset_cfg),
        "atlas_name": args.atlas_name,
        "sc_kinds": args.sc_kinds_resolved,
        "fc_kind": args.fc_kind,
        "split_ratio": args.split_ratio,
        "seed": args.seed,
        "labels": labels,
    }
    with open(os.path.join(combo_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=True)

    for label in labels:
        args.label_type = label
        label_output_dir = os.path.join(combo_dir, "label_{}".format(sanitize_name(label)))
        os.makedirs(label_output_dir, exist_ok=True)
        with open(os.path.join(label_output_dir, "label_name.txt"), "w", encoding="utf-8") as f:
            f.write(label + "\n")

        dt = dataset(
            dsType=args.dataset,
            labelType=args.label_type,
            use_dataset_cfg=args.use_dataset_cfg,
            dataset_name=args.dataset_name,
            atlas_name=args.atlas_name,
            sc_kind=args.sc_kind,
            sc_kinds=args.sc_kinds,
            fc_kind=args.fc_kind,
        )
        dt.setsubset(
            labelType=args.label_type,
            labeldim=args.hidden_dimension,
            split_ratio=args.split_ratio,
            create_val=True,
        )
        trainloader = dt.train_dataloader(batchsize=args.batch)
        
        # 根据参数控制是否启用标签规范化
        if args.no_normalize_labels or not args.normalize_labels:
            # 禁用规范化：mean=0, std=1（即不进行变换）
            lb_mean = torch.tensor(0.0, dtype=torch.float64)
            lb_std = torch.tensor(1.0, dtype=torch.float64)
            age_scale = 1.0
            print("Label/Age normalization: DISABLED (using raw values)")
        else:
            # 启用规范化：使用训练集的zscore
            train_labels = torch.tensor([data[1] for data in dt.train_dataset], dtype=torch.float64)
            lb_mean = train_labels.mean().to(device)
            lb_std = train_labels.std().to(device)
            age_scale = 100.0  # 年龄除以100进行缩放
            print(f"Label/Age normalization: ENABLED (lb_mean={lb_mean.item():.4f}, lb_std={lb_std.item():.4f})")
        
        testloader = dt.test_dataloader()
        valloader = dt.val_dataloader()
        print("dataset is okay for label: {}".format(label))

        model = LGUNet_rela(args).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.l2_penalty
        )
        
        # 创建动态学习率调度器（如果启用）
        dynamic_scheduler = None
        if args.use_dynamic_lr:
            dynamic_scheduler = DynamicLearningRateScheduler(
                optimizer=optimizer,
                base_lr=args.learning_rate,
                min_lr=args.min_lr,
                patience=args.lr_patience,
                factor=args.lr_factor,
                warmup_epochs=args.warmup_epochs,
            )
            print(f"Dynamic LR Scheduler ENABLED: base_lr={args.learning_rate}, "
                  f"min_lr={args.min_lr}, patience={args.lr_patience}, factor={args.lr_factor}")
            # 禁用原来的MultiStepLR，使用动态调度器
            scheduler = None
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[30, 60, 80], gamma=0.5
            )

        # (這裡是你原本 main 函數結尾的調用部分，請替換成下面這樣)
        train(args, model, trainloader, valloader, optimizer, scheduler, device, label_output_dir, 
              lb_mean, lb_std, age_scale, use_dynamic_lr=args.use_dynamic_lr, dynamic_scheduler=dynamic_scheduler)
        evaluate(args, testloader, device, label_output_dir, lb_mean, lb_std, age_scale)
        explain_model(args, testloader, device, label_output_dir, lb_mean, lb_std, age_scale)


if __name__ == "__main__":
    main()
