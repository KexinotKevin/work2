#!/bin/bash

conda activate gnn_work2
# 使用 conda run 直接执行，避免 conda activate 警告
conda run -n gnn_work2 python compute_graphs.py \
    --combo_dir debug_results/debug_final_restore_all_labels/20260404_071153/S1200/atlas_schaefer200_S1__sc_FA-fiber_count__fc_pcc_rest/split_70_15_15/seed_42/ \
    --atlas schaefer200_S1 \
    --plots_dir plots/debug_results/debug_final_restore_all_labels/20260404_071153/ 