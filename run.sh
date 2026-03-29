#!/bin/bash

PYTHONPATH="/home/shulab/anaconda3/envs/gnn_gpu/bin/python"
$PYTHONPATH run.py --use_dataset_cfg \
    --num_epochs 100 \
    --dataset_name S1200 \
    --atlas_name bna246 \
    --sc_kinds FA fiber_count \
    --fc_kind pcc_rest \
    --label_types 'CogFluidComp_Unadj','CogFluidComp_AgeAdj','CogEarlyComp_Unadj','CogEarlyComp_AgeAdj','CogTotalComp_Unadj','CogTotalComp_AgeAdj','CogCrystalComp_Unadj','CogCrystalComp_AgeAdj' \
    --output_root ./results

# replace <combo_dir> with the printed "result combo_dir" path from run.py
# python compute_graphs.py --combo_dir <combo_dir>
