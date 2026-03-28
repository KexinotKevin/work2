import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from dataset import dataset
from model_r import LGUNet_rela


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


def build_combo_dir(args):
    sc_tag = "-".join([sanitize_name(x) for x in args.sc_kinds_resolved])
    fc_tag = sanitize_name(args.fc_kind)
    combo_name = "atlas_{}__sc_{}__fc_{}".format(sanitize_name(args.atlas_name), sc_tag, fc_tag)
    return os.path.join(
        args.output_root,
        sanitize_name(args.dataset_name if args.use_dataset_cfg else args.dataset),
        combo_name,
        split_tag(args.split_ratio),
        "seed_{}".format(args.seed),
    )


def train(args, model, trainloader, valloader, optimizer, scheduler, device, label_output_dir):
    train_loss = []
    val_loss = []
    tmp_train = os.path.join(label_output_dir, "bt_tmp.pth")
    tmp_val = os.path.join(label_output_dir, "bv_tmp.pth")
    remove_pattern(os.path.join(label_output_dir, "bt_tmp*.pth"))
    remove_pattern(os.path.join(label_output_dir, "bv_tmp*.pth"))

    for epoch in range(args.num_epochs):
        t = time.time()
        train_loss_tmp = []
        model.train()
        for g_data, lb_data in trainloader:
            g_data = g_data.to(device)
            lb_data = lb_data.to(device)
            optimizer.zero_grad()
            lb_pred = model(g_data, lb_data, g_data.batch).squeeze(-1)
            loss = abs(lb_pred - lb_data).mean()
            loss.backward()
            optimizer.step()
            train_loss_tmp.append(loss.item())
        scheduler.step()
        train_loss_v = sum(train_loss_tmp) / len(train_loss_tmp)

        model.eval()
        val_loss_tmp = []
        with torch.no_grad():
            for g_data, lb_data in valloader:
                g_data = g_data.to(device)
                lb_data = lb_data.to(device)
                lb_pred = model(g_data, lb_data, g_data.batch).squeeze(-1)
                loss = abs(lb_pred - lb_data).mean()
                val_loss_tmp.append(loss.item())
        val_loss_v = sum(val_loss_tmp) / len(val_loss_tmp)

        train_loss.append(train_loss_v)
        val_loss.append(val_loss_v)
        print(
            "Epoch: {:04d}".format(epoch + 1),
            "loss_train: {:.4f}".format(train_loss_v),
            "loss_val: {:.4f}".format(val_loss_v),
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

    df = pd.DataFrame(columns=["train_loss", "val_loss"])
    df["train_loss"] = train_loss
    df["val_loss"] = val_loss
    df.to_csv(os.path.join(label_output_dir, "loss.csv"), index=False)


def evaluate(args, testloader, device, label_output_dir):
    model_t = torch.load(
        os.path.join(label_output_dir, "best_validation.pth")
    ).to(device)
    model_t.eval()

    from sklearn.metrics import r2_score, root_mean_squared_error
    from scipy.stats import pearsonr

    total_rmse = []
    total_r2 = []
    total_p = []
    for _ in range(args.test_repeat):
        with torch.no_grad():
            lb_p = []
            lb_t = []
            for g_test, lb_test in testloader:
                g_test = g_test.to(device)
                lb_test = lb_test.to(device)
                lb_pred = model_t(g_test, lb_test, g_test.batch).squeeze(-1)
                lb_test = lb_test.cpu().numpy()
                lb_pred = lb_pred.cpu().numpy()
                lb_t.append(lb_test)
                lb_p.append(lb_pred)
            lb_test = np.array(lb_t).squeeze()
            lb_pred = np.array(lb_p).squeeze()
            rmse_score = root_mean_squared_error(lb_test, lb_pred)
            r_square = r2_score(lb_test, lb_pred)
            p_corr = pearsonr(lb_test, lb_pred)
            total_rmse.append(rmse_score)
            total_r2.append(r_square)
            total_p.append(p_corr)

    df2 = pd.DataFrame(columns=["repeat_rmse", "repeat_r2", "pearson_corr"])
    df2["repeat_rmse"] = total_rmse
    df2["repeat_r2"] = total_r2
    df2["pearson_corr"] = total_p
    df2.to_csv(
        os.path.join(label_output_dir, "test.csv"),
        index=False,
    )


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
    parser.add_argument("--depth", type=float, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--pool_ratio", type=float, nargs="+", default=[0.5, 0.8, 0.5])

    parser.add_argument("--split_ratio", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_repeat", type=int, default=11)
    parser.add_argument("--output_root", type=str, default="./results")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
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

    combo_dir = build_combo_dir(args)
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
        testloader = dt.test_dataloader()
        valloader = dt.val_dataloader()
        print("dataset is okay for label: {}".format(label))

        model = LGUNet_rela(args).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.l2_penalty
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 20, 25], gamma=0.5
        )

        train(args, model, trainloader, valloader, optimizer, scheduler, device, label_output_dir)
        evaluate(args, testloader, device, label_output_dir)


if __name__ == "__main__":
    main()
