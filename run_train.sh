#!/bin/bash

# conda activate gnn_work2
python run.py --use_dataset_cfg \
    --num_epochs 100 \
    --dataset_name S1200 \
    --atlas_name bna246 \
    --sc_kinds FA fiber_length \
    --fc_kind pcc_rest \
    --label_types 'CogFluidComp_Unadj','CogEarlyComp_Unadj','CogTotalComp_Unadj','CogCrystalComp_Unadj' \
    --output_root ./results_full \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --depth 2

# python run.py --use_dataset_cfg \
#     --num_epochs 3 \
#     --dataset_name ABCD \
#     --atlas_name bna246 \
#     --sc_kinds FA fiber_count \
#     --fc_kind pcc_rest \
#     --label_types 'CogFluidComp_Unadj','CogFluidComp_AgeAdj','CogEarlyComp_Unadj','CogEarlyComp_AgeAdj','CogTotalComp_Unadj','CogTotalComp_AgeAdj','CogCrystalComp_Unadj','CogCrystalComp_AgeAdj' \
#     --output_root ./results

# replace <combo_dir> with the printed "result combo_dir" path from run.py
# python compute_graphs.py --combo_dir <combo_dir>

# S1200： 'CogFluidComp_Unadj','CogEarlyComp_Unadj','CogTotalComp_Unadj','CogCrystalComp_Unadj'
#ABCD: 'nihtbx_fluidcomp_uncorrected','nihtbx_cryst_uncorrected','nihtbx_totalcomp_uncorrected'