#!/bin/bash

set -e
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate gnn_work2

python ablation/generate_table.py \
    --ablation_exp ablation_results/ablation_hcd/20260407_021029 \
    --ours_exp results/hcd_all/20260406_122112/HCD/atlas_bna246__sc_fiber_count__fc_pcc_rest/split_70_15_15/seed_42

# 指定自定义 Ours 路径
python ablation/generate_table_part2.py \
    --exp_dir ablation_results/ablation_hcd/20260407_030533 \
    --ours_exp results/hcd_all/20260406_122112/HCD/atlas_bna246__sc_fiber_count__fc_pcc_rest/split_70_15_15/seed_42