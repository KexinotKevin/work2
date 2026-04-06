#!/usr/bin/env bash
set -e

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate gnn_work2

export PYTHONPATH="$(pwd)/../../":$PYTHONPATH

# 用法: ./generate_summary.sh 20260405_144555 [base_dir]
TIMESTAMP="${1:-}"
BASE_DIR="${2:-../results/debug_baseline}"

if [ -z "$TIMESTAMP" ]; then
    echo "用法: $0 <timestamp> [base_dir]"
    echo "示例: $0 20260405_144555"
    echo "示例: $0 20260405_144555 ../results/debug_baseline"
    exit 1
fi

python generate_summary.py \
    --timestamp "${TIMESTAMP}" \
    --base_dir "${BASE_DIR}"
