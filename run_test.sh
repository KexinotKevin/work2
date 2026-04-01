#!/bin/bash

conda activate gnn_work2
python test.py \
    --model_path results_full/20260331_154936/S1200/atlas_bna246__sc_FA-fiber_length__fc_pcc_rest/split_70_15_15/seed_42/label_CogFluidComp_Unadj/best_validation.pth \
    --label_type "CogFluidComp_Unadj" \
    --dataset_name S1200 \
    --atlas_name bna246 \
    --sc_kinds FA fiber_length \
    --fc_kind pcc_rest