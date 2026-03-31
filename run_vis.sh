#!/bin/bash

# 使用 conda run 直接执行，避免 conda activate 警告
conda run -n gnn_work2 python compute_graphs.py \
    --combo_dir ./results/20260331_081940/S1200/atlas_bna246__sc_FA-fiber_count__fc_pcc_rest/split_70_15_15/seed_42/ \
    --atlas bna246\
    --plots_dir plots/20260331_081940/S1200/atlas_bna246__sc_FA-fiber_count__fc_pcc_rest/split_70_15_15/seed_42/\
    --label 'CogFluidComp_Unadj'
