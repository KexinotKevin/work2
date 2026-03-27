import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import dataset
from runner import BrainRunner


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


def parse_args():
    parser = argparse.ArgumentParser(description="LG-BrainUNet main entry")
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
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    args.sc_kinds_resolved = (
        args.sc_kinds
        if args.sc_kinds is not None
        else [x.strip() for x in str(args.sc_kind).split(",") if x.strip()]
    )
    labels = [x.strip() for x in str(args.label_types).split(",") if x.strip()]
    if not labels:
        labels = [args.label_type]

    pl.seed_everything(args.seed, workers=True)
    torch.set_default_dtype(torch.float64)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
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
        "trainer_accelerator": accelerator,
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
            labelType=label,
            use_dataset_cfg=args.use_dataset_cfg,
            dataset_name=args.dataset_name,
            atlas_name=args.atlas_name,
            sc_kind=args.sc_kind,
            sc_kinds=args.sc_kinds,
            fc_kind=args.fc_kind,
        )
        dt.setsubset(
            labelType=label,
            labeldim=args.hidden_dimension,
            split_ratio=args.split_ratio,
            create_val=True,
        )
        trainloader = dt.train_dataloader(batchsize=args.batch)
        valloader = dt.val_dataloader()
        testloader = dt.test_dataloader()
        print("dataset is okay for label: {}".format(label))

        ckpt_train = ModelCheckpoint(
            dirpath=label_output_dir,
            filename="best_train",
            monitor="train_loss",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=False,
        )
        ckpt_val = ModelCheckpoint(
            dirpath=label_output_dir,
            filename="best_validation",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=False,
        )
        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            accelerator=accelerator,
            devices=1,
            callbacks=[ckpt_train, ckpt_val],
            logger=False,
            enable_progress_bar=True,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            deterministic=True,
        )
        runner = BrainRunner(args)
        trainer.fit(runner, train_dataloaders=trainloader, val_dataloaders=valloader)
        if not ckpt_train.best_model_path or not ckpt_val.best_model_path:
            raise RuntimeError("checkpoint was not saved as expected.")

        loss_df = pd.DataFrame(
            {
                "train_loss": runner.train_loss_history,
                "val_loss": runner.val_loss_history[: len(runner.train_loss_history)],
            }
        )
        loss_df.to_csv(os.path.join(label_output_dir, "loss.csv"), index=False)

        best_train_runner = BrainRunner.load_from_checkpoint(ckpt_train.best_model_path, args=args)
        torch.save(best_train_runner.model, os.path.join(label_output_dir, "best_train.pth"))

        best_val_ckpt = ckpt_val.best_model_path
        best_runner = BrainRunner.load_from_checkpoint(best_val_ckpt, args=args)
        torch.save(best_runner.model, os.path.join(label_output_dir, "best_validation.pth"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_runner = best_runner.to(device)
        best_runner.eval()

        from scipy.stats import pearsonr
        from sklearn.metrics import r2_score, root_mean_squared_error

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
                    lb_pred = best_runner(g_test, lb_test)
                    lb_t.append(lb_test.cpu().numpy())
                    lb_p.append(lb_pred.cpu().numpy())
                lb_test_np = np.array(lb_t).squeeze()
                lb_pred_np = np.array(lb_p).squeeze()
                total_rmse.append(root_mean_squared_error(lb_test_np, lb_pred_np))
                total_r2.append(r2_score(lb_test_np, lb_pred_np))
                total_p.append(pearsonr(lb_test_np, lb_pred_np))

        test_df = pd.DataFrame(
            {"repeat_rmse": total_rmse, "repeat_r2": total_r2, "pearson_corr": total_p}
        )
        test_df.to_csv(os.path.join(label_output_dir, "test.csv"), index=False)


if __name__ == "__main__":
    main()
