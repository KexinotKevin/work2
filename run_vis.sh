#!/bin/bash

conda activate gnn_work2
# 使用 conda run 直接执行，避免 conda activate 警告
conda run -n gnn_work2 python compute_graphs.py \
    --combo_dir results_full/20260331_154936/S1200/atlas_bna246__sc_FA-fiber_length__fc_pcc_rest/split_70_15_15/seed_42 \
    --atlas bna246\
    --plots_dir plots_full/20260331_154936/S1200/atlas_bna246__sc_FA-fiber_length__fc_pcc_rest/split_70_15_15/seed_42 