#!/bin/bash

conda activate gnn_work2
python compute_graphs.py \
    --combo_dir results/20260331_003241/S1200/atlas_bna246__sc_FA-fiber_count__fc_pcc_rest/split_70_15_15/seed_42/ \
    --atlas bna246\
    --plots_dir plots/20260331_003241/S1200/atlas_bna246__sc_FA-fiber_count__fc_pcc_rest/split_70_15_15/seed_42/\
    --label 'CogFluidComp_Unadj'
