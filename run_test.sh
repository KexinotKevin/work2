#!/bin/bash

/home/shulab/anaconda3/envs/gnn_gpu/bin/python test.py \
    --model_path results/20260328_165042/S1200/atlas_bna246__sc_FA-fiber_count__fc_pcc_rest/split_70_15_15/seed_42/label_CogFluidComp_Unadj/best_validation.pth \
    --label_type "CogFluidComp_Unadj" \
    --dataset_name S1200 \
    --atlas_name bna246 \
    --sc_kinds FA fiber_count \
    --fc_kind pcc_rest