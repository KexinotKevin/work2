#!/bin/bash

conda activate gnn_work2

# 示例1：在原数据集(S1200)的测试集上进行测试 (同分布评估)
python test.py \
    --model_path debug_results/debug_adamw/20260404_025222/S1200/atlas_schaefer200_S1__sc_FA-fiber_count__fc_pcc_rest/split_70_15_15/seed_42/label_CogFluidComp_Unadj/best_validation.pth \
    --label_type "CogFluidComp_Unadj" \
    --dataset_name S1200 \
    --atlas_name bna246 \
    --sc_kinds FA fiber_count \
    --fc_kind pcc_rest \
    --split_ratio 0.7 0.15 0.15 \
    --seed 42 \
    --partition test \
    --test_repeat 5

# 示例2：跨数据集测试（例如：用上面训练好的S1200模型，在ABCD的全部数据上做推理！）
# python test.py \
#     --model_path results_full/20260331_154936/S1200/atlas_bna246__sc_FA-fiber_length__fc_pcc_rest/split_70_15_15/seed_42/label_CogFluidComp_Unadj/best_validation.pth \
#     --label_type "CogFluidComp_Unadj" \
#     --dataset_name ABCD \
#     --atlas_name bna246 \
#     --sc_kinds FA fiber_length \
#     --fc_kind pcc_rest \
#     --partition all \
#     --test_repeat 1
