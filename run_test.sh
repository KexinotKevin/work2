#!/bin/bash
set -euo pipefail

# 非交互 bash 中需先 source conda.sh，否则 `conda activate` 会报 conda init 相关错误
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi
conda activate gnn_work2

# 跨数据集时：--label_type 必须是「目标数据集」分数表中的列名；训练时用的列名从 checkpoint 旁 label_name.txt 自动读取。
# 示例1：HCD 上训练的模型在 ABCD 上测（目标列为 ABCD 的 nihtbx_fluidcomp_uncorrected）
python test.py \
    --model_path debug_results/hcd_mat_test/20260406_053055/HCD/atlas_bna246__sc_fiber_count__fc_pcc_rest/split_70_15_15/seed_42/label_nih_fluidcogcomp_unadjusted/best_validation.pth \
    --label_type "CogFluidComp_Unadj" \
    --dataset_name S1200 \
    --atlas_name bna246 \
    --sc_kinds fiber_count \
    --fc_kind pcc_rest \
    --seed 42 \
    --partition all \
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
