"""
Aggregate HCD experiment results and generate metrics tables in the style of baseline_metrics.csv.

Outputs:
  - baseline_metrics_aggregated.csv   (CSV, same format as baseline_metrics.csv)
  - baseline_metrics_aggregated.tex    (LaTeX three-line table)

Usage:
  python baseline_models/pipelines/generate_metrics_table.py

Data sources:
  Baseline experiments:
    baseline_models/results/baseline_hcd/{exp_id}/{exp_id}/{model_dir}/
      bna246__sc_fiber_count__fc_pcc_rest/split_80_10_10/seed_*/label_*/test.csv
  Ours experiment:
    The script recursively finds seed_*/label_*/test.csv under each ours_roots entry,
    so you can pass either the full path or a parent directory.
    Example:
      --ours_roots results/hcd_all/20260406_122112
      --ours_roots results/hcd_all/20260406_122112/HCD/atlas_bna246__sc_fiber_count__fc_pcc_rest/split_70_15_15/seed_42
    Both work equivalently.
"""

import os
import glob
import pandas as pd
import numpy as np
import argparse
import time
import shutil

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Model taxonomy & label mapping
# ──────────────────────────────────────────────────────────────────────────────

MODEL_TAXONOMY = [
    # Vanilla GNN
    ("Vanilla GNN",               "GCN",               "SC",                "GCN_SC"),
    ("Vanilla GNN",               "GCN",               "FC",                "GCN_FC"),
    ("Vanilla GNN",               "GCN",               'concat([SC FC])',  "GCN_SC_FC"),
    ("Vanilla GNN",               "GAT",               "SC",                "GAT_SC"),
    ("Vanilla GNN",               "GAT",               "FC",                "GAT_FC"),
    ("Vanilla GNN",               "GAT",               'concat([SC FC])',  "GAT_SC_FC"),
    # Advanced GNN
    ("Advanced GNN",              "GraphSAGE",         "SC",                "SAGE_SC"),
    ("Advanced GNN",              "GraphSAGE",         "FC",                "SAGE_FC"),
    ("Advanced GNN",              "GraphSAGE",         'concat([SC FC])', "SAGE_SC_FC"),
    ("Advanced GNN",              "Graph Transformer", "SC",                "GraphTransformer_SC"),
    ("Advanced GNN",              "Graph Transformer", "FC",               "GraphTransformer_FC"),
    ("Advanced GNN",              "Graph Transformer", "SC-FC",            "GraphTransformer_SC_FC"),
    ("Advanced GNN",              "RGCN",              "SC-FC",             "RelGNN_SC_FC"),
    # Brain Graph Specific Designs
    ("Brain Graph Specific Designs", "BrainNetCNN",    "SC",                "BrainNetCNN_SC"),
    ("Brain Graph Specific Designs", "BrainNetCNN",    "FC",                "BrainNetCNN_FC"),
    ("Brain Graph Specific Designs", "BrainNetCNN",    "SC-FC",             "BrainNetCNN_SC_FC"),
    ("Brain Graph Specific Designs", "BrainGNN",       "SC",                "BrainGNN_SC"),
    ("Brain Graph Specific Designs", "BrainGNN",       "FC",                "BrainGNN_FC"),
    ("Brain Graph Specific Designs", "BrainGNN",       'concat([SC FC])',  "BrainGNN_SC_FC"),
    ("Brain Graph Specific Designs", "BrainNetworkTransformer", "SC",        "BNT_SC"),
    ("Brain Graph Specific Designs", "BrainNetworkTransformer", "FC",        "BNT_FC"),
    ("Brain Graph Specific Designs", "BrainRGIN",      "SC",                "BrainRGIN_SC"),
    ("Brain Graph Specific Designs", "BrainRGIN",      "FC",                "BrainRGIN_FC"),
    ("Brain Graph Specific Designs", "BrainRGIN",     "SC-FC",             "BrainRGIN_SC_FC"),
    # Ours
    ("Ours",                      "Ours",              "SC-FC",             "Ours"),
]

# display_name -> label_dir_name
LABEL_MAP = {
    "Fluid Intelligence":             "nih_fluidcogcomp_unadjusted",
    "Crystal Intelligence":             "nih_crycogcomp_unadjusted",
    "Total Composite Intelligence":     "nih_totalcogcomp_unadjusted",
    "Early Composite Intelligence":     "nih_eccogcomp_unadjusted",
}

LABEL_ORDER = [
    "Fluid Intelligence",
    "Crystal Intelligence",
    "Total Composite Intelligence",
    "Early Composite Intelligence",
]

# metrics in order, per label group
METRICS = ["RMSE", "MAE", "$R^2$", "Pearson $r$", "CCC"]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Path resolution
# ──────────────────────────────────────────────────────────────────────────────

def find_test_csvs_recursive(root, label_dir):
    """
    Recursively find test.csv files for a given label_dir under root.
    Handles both:
      root/seed_*/label_{label_dir}/test.csv
      root/.../HCD/.../split_*/seed_*/label_{label_dir}/test.csv
    Returns a list of absolute paths.
    """
    pattern = os.path.join(root, "**", f"label_{label_dir}", "test.csv")
    return [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]


def resolve_test_csvs_for_ours(roots, label_dir):
    """
    For Ours, search all roots recursively and return all matching test.csv paths.
    This handles arbitrary nesting levels.
    """
    results = []
    for root in roots:
        results.extend(find_test_csvs_recursive(root, label_dir))
    return results


def resolve_test_csv_for_baseline(roots, modality_key, label_dir):
    """
    For baseline models, use the fixed path structure.
    """
    for root in roots:
        pattern = os.path.join(
            root, modality_key,
            "bna246__sc_fiber_count__fc_pcc_rest",
            "split_80_10_10",
            "seed_*",
            f"label_{label_dir}",
            "test.csv"
        )
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Data reading & aggregation
# ──────────────────────────────────────────────────────────────────────────────

def read_and_aggregate(test_csv_paths):
    """
    Read one or more test.csv files, compute the grand mean across all
    repeat rows in all files.
    test_csv_paths: list of absolute paths (for Ours, multiple seeds/experiments)
                    or single-item list (for baseline, single seed)
    Returns a dict of metric_name -> mean value, or None on failure.
    """
    if not test_csv_paths:
        return None

    all_rows = []
    for p in test_csv_paths:
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p)
            df = df.dropna(how="all")
            if not df.empty:
                all_rows.append(df)
        except Exception:
            continue

    if not all_rows:
        return None

    # Concatenate all repeat rows from all files
    combined = pd.concat(all_rows, ignore_index=True)
    return {
        "RMSE":        combined["repeat_rmse"].mean(),
        "MAE":         combined["repeat_mae"].mean(),
        "$R^2$":       combined["repeat_r2"].mean(),
        "Pearson $r$": combined["pearson_corr"].mean(),
        "CCC":         combined["repeat_ccc"].mean(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Table assembly
# ──────────────────────────────────────────────────────────────────────────────

def build_row(test_csv_by_label):
    """
    Build a single data row metrics section.
    For each label: RMSE, MAE, $R^2$, Pearson $r$, CCC
    """
    row = []
    for label in LABEL_ORDER:
        metrics = test_csv_by_label.get(label)
        if metrics:
            row.extend([
                metrics["RMSE"],
                metrics["MAE"],
                metrics["$R^2$"],
                metrics["Pearson $r$"],
                metrics["CCC"],
            ])
        else:
            row.extend(["", "", "", "", ""])
    return row


# ──────────────────────────────────────────────────────────────────────────────
# 5.  CSV generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_csv(out_path, header_rows, data_rows):
    lines = []
    for row in header_rows + data_rows:
        lines.append(",".join(str(v) for v in row))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  LaTeX generation
# ──────────────────────────────────────────────────────────────────────────────

GU_LATEX = {
    "SC":                "SC",
    "FC":                "FC",
    "SC-FC":             "SC-FC",
    'concat([SC FC])':  'concat([SC FC])',
}


def generate_latex(out_path, header_rows, data_rows, timestamp):
    n_labels = len(LABEL_ORDER)
    n_metrics = len(METRICS)  # 5
    n_cols = 1 + 1 + 1 + n_labels * n_metrics   # type + model + graph + metrics

    col_spec = "l" + ("c" * (n_cols - 1))

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Main Results on HCD dataset ({timestamp})}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # ── Header row 1: group labels ───────────────────────────────────────────
    hr1 = [r"\textbf{Model Type}", r"\textbf{Model}", r"\textbf{Graph}"]
    for label in LABEL_ORDER:
        short = label.replace(" Composite", "").replace(" Intelligence", "")
        hr1.append(rf"\multicolumn{{{n_metrics}}}{{c}}{{\textbf{{{short}}}}}")
    lines.append(" & ".join(hr1) + r" \\")
    lines.append(r"\midrule")

    # ── Header row 2: metric names ─────────────────────────────────────────
    hr2 = [r"", r"", r"Usage"]
    for _ in LABEL_ORDER:
        for m in METRICS:
            m_disp = m.replace("$R^2$", r"R$^2$")
            hr2.append(rf"\textbf{{{m_disp}}}")
    lines.append(" & ".join(hr2) + r" \\")

    # ── Data rows ─────────────────────────────────────────────────────────
    def fmt(v):
        if v == "" or v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        return f"{v:.4f}"

    for row in data_rows:
        model_type, model_name = row[0], row[1]
        graph_usage = row[2]
        parts = [
            model_type or "",
            model_name or "",
            GU_LATEX.get(graph_usage, graph_usage),
        ]
        for val in row[3:]:
            parts.append(fmt(val))
        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\label{tab:baseline_hcd}")
    lines.append(r"\end{table}")

    content = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content + "\n")
    return content


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aggregate HCD metrics tables")
    parser.add_argument("--baseline_roots", nargs="+",
                        default=[
                            "baseline_models/results/baseline_hcd/20260406_140020/20260406_140020",
                            "baseline_models/results/baseline_hcd/20260406_085531/20260406_085531",
                        ],
                        help="Root directories for baseline experiment results")
    parser.add_argument("--ours_roots", nargs="+",
                        default=[
                            "results/hcd_all/20260406_122112",
                            "results/hcd_all/20260406_115323",
                        ],
                        help="Root directories for Ours experiment results (recursive search)")
    parser.add_argument("--output_dir",
                        default="baseline_models/baseline_tables",
                        help="Directory to save output CSV and TEX files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # ── Collect metrics ──────────────────────────────────────────────────────
    all_results = []  # list of dicts

    for model_type, model_name, graph_usage, modality_key in MODEL_TAXONOMY:
        test_csv_by_label = {}  # display_name -> {RMSE, MAE, ...}

        for label_col, label_dir in LABEL_MAP.items():
            if modality_key == "Ours":
                paths = resolve_test_csvs_for_ours(args.ours_roots, label_dir)
                agg = read_and_aggregate(paths)
            else:
                path = resolve_test_csv_for_baseline(
                    args.baseline_roots, modality_key, label_dir
                )
                agg = read_and_aggregate([path] if path else [])

            if agg is not None:
                test_csv_by_label[label_col] = agg

        row = [model_type, model_name, graph_usage]
        row.extend(build_row(test_csv_by_label))

        all_results.append({
            "model_type":  model_type,
            "model_name":  model_name,
            "graph_usage": graph_usage,
            "row":         row,
        })

    # ── Build header rows ────────────────────────────────────────────────────
    n_m = len(METRICS)   # 5
    row1 = ["Model Type", "Model", "Graph Usage"]
    for label in LABEL_ORDER:
        row1.append(label)
        row1.extend([""] * (n_m - 1))

    row2 = ["", "", ""]
    for _ in LABEL_ORDER:
        for m in METRICS:
            row2.append(m)

    header_rows = [row1, row2]
    data_rows = [r["row"] for r in all_results]

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, f"baseline_metrics_aggregated_{ts}.csv")
    generate_csv(csv_path, header_rows, data_rows)
    print(f"[OK] CSV saved → {csv_path}")

    # ── Save LaTeX ───────────────────────────────────────────────────────────
    tex_path = os.path.join(args.output_dir, f"baseline_metrics_aggregated_{ts}.tex")
    latex_content = generate_latex(tex_path, header_rows, data_rows, ts)
    print(f"[OK] LaTeX saved → {tex_path}")

    # ── Also save latest copies ──────────────────────────────────────────────
    latest_csv = os.path.join(args.output_dir, "baseline_metrics_aggregated.csv")
    latest_tex = os.path.join(args.output_dir, "baseline_metrics_aggregated.tex")
    shutil.copy(csv_path, latest_csv)
    shutil.copy(tex_path, latest_tex)
    print(f"[OK] Latest copies: {latest_csv}, {latest_tex}")

    # ── Summary ───────────────────────────────────────────────────────────────
    found = sum(1 for r in all_results if any(v != "" for v in r["row"][3:]))
    print(f"\n{found}/{len(all_results)} model entries have data.")

    print("\n" + "=" * 70)
    print("LaTeX TABLE PREVIEW:")
    print("=" * 70)
    print(latex_content)
    print("=" * 70)


if __name__ == "__main__":
    main()
