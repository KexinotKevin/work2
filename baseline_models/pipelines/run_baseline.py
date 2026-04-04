import argparse, os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from dataset import dataset
from baseline_models.models.vanilla_gnn import VanillaGCN, VanillaGAT, VanillaSAGE, VanillaRelGNN
from strategies import EarlyStopping


def transform_batch(g_data, args, device):
    """Transforms raw graph data into Baseline format without modifying load_data.py.
    
    This function reconstructs dense matrices from the compressed edge_attr and
    assigns them as node features. For RelGNN, edge attributes are preserved for
    dual-relation modeling (SC_FA and FC_pos). For GCN/GAT/SAGE, edge_attr is
    discarded and the models operate on topology + numerical node features only.
    """
    sc_dim = len(args.sc_kinds_resolved)
    x_list = []

    for i in range(g_data.num_graphs):
        mask = g_data.batch == i
        num_nodes = mask.sum().item()
        e_mask = g_data.batch[g_data.edge_index[0]] == i
        e_idx = g_data.edge_index[:, e_mask] - g_data.ptr[i]

        adj_sc = torch.zeros((num_nodes, num_nodes), device=device, dtype=g_data.edge_attr.dtype)
        adj_sc[e_idx[0], e_idx[1]] = g_data.edge_attr[e_mask, 0]

        adj_fc = torch.zeros((num_nodes, num_nodes), device=device, dtype=g_data.edge_attr.dtype)
        adj_fc[e_idx[0], e_idx[1]] = g_data.edge_attr[e_mask, sc_dim] - g_data.edge_attr[e_mask, sc_dim + 1]

        if args.model_type == "RelGNN":
            x_list.append(adj_fc)
        else:
            if args.modality == 'SC':
                x_list.append(adj_sc)
            elif args.modality == 'FC':
                x_list.append(adj_fc)
            elif args.modality == 'SC_FC':
                x_list.append(torch.cat([adj_sc, adj_fc], dim=-1))

    g_data.x = torch.cat(x_list, dim=0).to(device)

    if args.model_type == "RelGNN":
        g_data.edge_attr = g_data.edge_attr[:, [0, sc_dim]].to(device)
    else:
        g_data.edge_attr = None

    return g_data


def setup_logging(output_root, timestamp=None):
    """Setup logging to both terminal and file."""
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


def train_baseline(args, model, trainloader, valloader, optimizer, device, label_output_dir, lb_mean, lb_std, early_stopping=None):
    train_loss_history, val_loss_history = [], []
    best_val_epoch = 0
    epochs_trained = 0

    for epoch in range(args.num_epochs):
        t = time.time()
        model.train()
        train_loss_tmp = []

        for g_data, lb_data in trainloader:
            g_data, lb_data = g_data.to(device), lb_data.to(device)
            g_data = transform_batch(g_data, args, device)

            optimizer.zero_grad()
            out_cog, _, _ = model(g_data, None, g_data.batch)

            lb_data_norm = (lb_data - lb_mean) / (lb_std + 1e-8)
            loss_cog = F.mse_loss(out_cog.squeeze(-1), lb_data_norm)

            loss_cog.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
            optimizer.step()
            train_loss_tmp.append(loss_cog.item())

        train_loss_v = np.mean(train_loss_tmp)

        model.eval()
        val_loss_tmp = []
        with torch.no_grad():
            for g_data, lb_data in valloader:
                g_data, lb_data = g_data.to(device), lb_data.to(device)
                g_data = transform_batch(g_data, args, device)

                out_cog, _, _ = model(g_data, None, g_data.batch)
                lb_data_norm = (lb_data - lb_mean) / (lb_std + 1e-8)
                loss = F.mse_loss(out_cog.squeeze(-1), lb_data_norm)
                val_loss_tmp.append(loss.item())

        val_loss_v = np.mean(val_loss_tmp) if val_loss_tmp else float('inf')

        train_loss_history.append(train_loss_v)
        val_loss_history.append(val_loss_v)
        epochs_trained = epoch + 1

        if val_loss_v <= min(val_loss_history):
            best_val_epoch = epoch
            torch.save(model, os.path.join(label_output_dir, "best_validation.pth"))

        print(
            "Epoch: {:04d}".format(epoch + 1),
            "loss_train: {:.4f}".format(train_loss_v),
            "loss_val: {:.4f}".format(val_loss_v),
            "time: {:.4f}s".format(time.time() - t),
        )

        if early_stopping is not None:
            model_state = model.state_dict()
            should_stop = early_stopping(epoch, val_loss_v, model_state)
            if should_stop:
                print(f"\n>>> Early stopping at epoch {epoch + 1}.")
                print(f">>> Best validation loss at epoch {early_stopping.get_best_epoch() + 1}")
                early_stopping.restore_weights(model)
                torch.save(model, os.path.join(label_output_dir, "best_validation.pth"))
                break

    if early_stopping is None or not early_stopping.should_stop:
        print(f"\n>>> Training completed for all {epochs_trained} epochs.")

    df = pd.DataFrame({"train_loss": train_loss_history, "val_loss": val_loss_history})
    df.to_csv(os.path.join(label_output_dir, "loss.csv"), index=False)

    return epochs_trained, best_val_epoch


def evaluate_baseline(args, testloader, device, label_output_dir, lb_mean, lb_std, checkpoint_path=None):
    ckpt = checkpoint_path if checkpoint_path is not None else os.path.join(label_output_dir, "best_validation.pth")
    model_t = torch.load(ckpt, weights_only=False).to(device)
    model_t.eval()

    from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr

    total_rmse, total_mae, total_r2, total_p = [], [], [], []

    print(f"\n>>> Starting Evaluation for Label: {label_output_dir}")
    for _ in range(args.test_repeat):
        with torch.no_grad():
            lb_p, lb_t = [], []
            for g_test, lb_test in testloader:
                g_test = g_test.to(device)
                lb_test = lb_test.to(device)
                g_test = transform_batch(g_test, args, device)

                lb_pred, _, _ = model_t(g_test, None, g_test.batch)

                lb_pred_real = lb_pred.squeeze(-1) * lb_std + lb_mean
                lb_t.append(lb_test.cpu().numpy())
                lb_p.append(lb_pred_real.cpu().numpy())

            lb_test = np.array(lb_t).flatten()
            lb_pred = np.array(lb_p).flatten()

            total_rmse.append(root_mean_squared_error(lb_test, lb_pred))
            total_mae.append(mean_absolute_error(lb_test, lb_pred))
            total_r2.append(r2_score(lb_test, lb_pred))
            total_p.append(pearsonr(lb_test, lb_pred)[0])

    print(f"Test Results over {args.test_repeat} repeats:")
    print(f"  RMSE: {np.mean(total_rmse):.4f} ± {np.std(total_rmse):.4f}")
    print(f"  MAE:  {np.mean(total_mae):.4f} ± {np.std(total_mae):.4f}")
    print(f"  R2:   {np.mean(total_r2):.4f} ± {np.std(total_r2):.4f}")
    print(f"  Pearson r: {np.mean(total_p):.4f} ± {np.std(total_p):.4f}")

    df2 = pd.DataFrame({
        "repeat_rmse": total_rmse,
        "repeat_mae": total_mae,
        "repeat_r2": total_r2,
        "pearson_corr": total_p
    })
    df2.to_csv(os.path.join(label_output_dir, "test.csv"), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline GNN run entry")
    parser.add_argument("--model_type", type=str, default="GCN", choices=["GCN", "GAT", "SAGE", "RelGNN"])
    parser.add_argument("--modality", type=str, default="SC", choices=["SC", "FC", "SC_FC"])

    parser.add_argument("--dataset_class", type=str, default="dataset", choices=["dataset"])
    parser.add_argument("--dataset", type=str, default="HCD", choices=["HCP", "HCD"])
    parser.add_argument("--label_type", type=str, default="CogFluidComp_Unadj")
    parser.add_argument("--label_types", type=str, default="")

    parser.add_argument("--use_dataset_cfg", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="HCD")
    parser.add_argument("--atlas_name", type=str, default="bna246")
    parser.add_argument("--num_nodes", type=int, default=246, help="Atlas nodes count (bna=246, schaefer=200)")
    parser.add_argument("--sc_kind", type=str, default="FA")
    parser.add_argument("--sc_kinds", type=str, nargs="+", default=None)
    parser.add_argument("--fc_kind", type=str, default="pcc_rest")

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--l2_penalty", type=float, default=0.001)
    parser.add_argument("--input_dimension", type=int, default=246)
    parser.add_argument("--relation_num", type=int, default=2)
    parser.add_argument("--hidden_dimension", type=int, default=64)
    parser.add_argument("--output_dimension", type=int, default=1)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--split_ratio", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_repeat", type=int, default=11)
    parser.add_argument("--output_root", type=str, default="./baseline_results")

    parser.add_argument("--normalize_labels", action="store_true", default=True)
    parser.add_argument("--no_normalize_labels", action="store_true")

    parser.add_argument("--use_early_stopping", action="store_true", default=False)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--early_stopping_min_epochs", type=int, default=10)
    parser.add_argument("--early_stopping_restore_best", action="store_true", default=True)

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    setup_logging(args.output_root, timestamp)

    args.sc_kinds_resolved = args.sc_kinds if args.sc_kinds is not None else [x.strip() for x in str(args.sc_kind).split(",") if x.strip()]

    if args.model_type != "RelGNN" and args.modality == "SC_FC":
        args.input_dimension = args.num_nodes * 2
    else:
        args.input_dimension = args.num_nodes

    args.relation_num = 2 if args.model_type == "RelGNN" else 0

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
        "model_type": args.model_type,
        "modality": args.modality,
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "use_dataset_cfg": bool(args.use_dataset_cfg),
        "atlas_name": args.atlas_name,
        "num_nodes": args.num_nodes,
        "sc_kinds": args.sc_kinds_resolved,
        "fc_kind": args.fc_kind,
        "split_ratio": args.split_ratio,
        "seed": args.seed,
        "labels": labels,
        "input_dimension": args.input_dimension,
        "relation_num": args.relation_num,
    }
    with open(os.path.join(combo_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=True)

    for label in labels:
        args.label_type = label
        label_output_dir = os.path.join(combo_dir, "label_{}".format(sanitize_name(label)))
        os.makedirs(label_output_dir, exist_ok=True)
        with open(os.path.join(label_output_dir, "label_name.txt"), "w") as f:
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
            output_dir=label_output_dir,
        )
        dt.setsubset(
            labelType=args.label_type,
            labeldim=args.hidden_dimension,
            split_ratio=args.split_ratio,
            create_val=True,
        )

        if args.no_normalize_labels or not args.normalize_labels:
            lb_mean = torch.tensor(0.0, dtype=torch.float64)
            lb_std = torch.tensor(1.0, dtype=torch.float64)
            print("Label normalization: DISABLED")
        else:
            train_labels = torch.tensor([data[1] for data in dt.train_dataset], dtype=torch.float64)
            lb_mean = train_labels.mean().to(device)
            lb_std = train_labels.std().to(device)
            print(f"Label normalization: ENABLED (mean={lb_mean.item():.4f}, std={lb_std.item():.4f})")

        trainloader = dt.train_dataloader(batchsize=args.batch)
        valloader = dt.val_dataloader()
        testloader = dt.test_dataloader()
        print("dataset loaded for label: {}".format(label))

        model_map = {"GCN": VanillaGCN, "GAT": VanillaGAT, "SAGE": VanillaSAGE, "RelGNN": VanillaRelGNN}
        model = model_map[args.model_type](args).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.l2_penalty
        )

        early_stopping = None
        if args.use_early_stopping:
            early_stopping = EarlyStopping(
                patience=args.early_stopping_patience,
                min_delta=args.early_stopping_min_delta,
                mode='min',
                restore_best_weights=args.early_stopping_restore_best,
                min_epochs=args.early_stopping_min_epochs,
                verbose=True,
            )
            print(f"Early Stopping ENABLED: patience={args.early_stopping_patience}")

        train_baseline(args, model, trainloader, valloader, optimizer, device, label_output_dir, lb_mean, lb_std, early_stopping)
        evaluate_baseline(args, testloader, device, label_output_dir, lb_mean, lb_std)


if __name__ == "__main__":
    main()
