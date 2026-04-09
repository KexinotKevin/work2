#!/bin/bash

set -e
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate gnn_work2

python baseline_models/pipelines/generate_metrics_table.py \
    --baseline_roots baseline_models/results/baseline_hcd/20260406_140020/20260406_140020 \
    --ours_roots results/hcd_all/20260406_122112 \
    --output_dir baseline_models/baseline_tables