#!/bin/bash

PYTHONPATH="/home/shulab/anaconda3/envs/gnn_gpu/bin/python"
$PYTHONPATH compute_graphs.py \
    --combo_dir results/20260328_165042/S1200/atlas_bna246__sc_FA-fiber_count__fc_pcc_rest/split_70_15_15/seed_42/ \
    --atlas bna246\
    --plots_dir plots/20260328_165042/S1200/atlas_bna246__sc_FA-fiber_count__fc_pcc_rest/split_70_15_15/seed_42/\
    --label "CogFluidComp_Unadj"
