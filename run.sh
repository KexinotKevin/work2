#!/bin/bash

conda activate gnn_gpu
python run.py --use_dataset_cfg \ 
    --dataset_name S1200 \
    --atlas_name bna246 \
    --sc_kinds FA fiber_count \
    --fc_kind pcc_rest \
    --label_type nih_fluidcogcomp_unadjusted
