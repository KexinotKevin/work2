#!/bin/bash

conda activate gnn_gpu
python run.py --use_dataset_cfg \
    --dataset_name S1200 \
    --atlas_name bna246 \
    --sc_kinds FA fiber_count \
    --fc_kind pcc_rest \
    --label_types "nih_fluidcogcomp_unadjusted,nih_crycogcomp_unadjusted,nih_totalcogcomp_unadjusted" \
    --output_root ./results

# replace <combo_dir> with the printed "result combo_dir" path from run.py
# python compute_graphs.py --combo_dir <combo_dir>
