#!/usr/bin/env bash
# 用 `bash run_train.sh` 运行时是非交互 shell，必须先加载 conda 的 shell 集成，否则会出现:
#   CondaError: Run 'conda init' before 'conda activate'
set -e
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate gnn_work2

# ============================================
# 【规范化控制说明】
# - 默认启用标签和年龄的zscore规范化
# - 如需禁用规范化，添加 --no_normalize_labels 参数
# ============================================

# 【使用动态学习率调整】
# --use_dynamic_lr: 启用动态学习率调度器
# --lr_patience: loss plateau检测的耐心值（默认10）
# --lr_factor: plateau后学习率衰减因子（默认0.5）
# --min_lr: 最小学习率（默认1e-6）
# --warmup_epochs: 预热epoch数（默认5）

python run.py --use_dataset_cfg \
    --num_epochs 50 \
    --dataset_name S1200 \
    --atlas_name schaefer200_S1 \
    --sc_kinds FA fiber_count \
    --fc_kind pcc_rest \
    --label_types 'CogFluidComp_Unadj','CogEarlyComp_Unadj','CogTotalComp_Unadj','CogCrystalComp_Unadj' \
    --output_root ./debug_results/test_thr_avail \
    --depth 2 \
    --use_early_stopping \
    --use_dynamic_lr \
    --lr_patience 10 \
    --lr_factor 0.5 \
    --min_lr 1e-6 \
    --warmup_epochs 5

# 【禁用规范化且使用动态学习率示例】
# python run.py --use_dataset_cfg \
#     --num_epochs 100 \
#     --dataset_name S1200 \
#     --atlas_name bna246 \
#     --sc_kinds FA fiber_length \
#     --fc_kind pcc_rest \
#     --label_types 'CogFluidComp_Unadj','CogEarlyComp_Unadj','CogTotalComp_Unadj','CogCrystalComp_Unadj' \
#     --output_root ./results_dyn_lr \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --depth 2 \
#     --no_normalize_labels \
#     --use_dynamic_lr \
#     --lr_patience 15 \
#     --lr_factor 0.3 \
#     --min_lr 1e-7 \
#     --warmup_epochs 10

# 【原始静态学习率（无动态调整）】
# python run.py --use_dataset_cfg \
#     --num_epochs 100 \
#     --dataset_name S1200 \
#     --atlas_name bna246 \
#     --sc_kinds FA fiber_length \
#     --fc_kind pcc_rest \
#     --label_types 'CogFluidComp_Unadj','CogEarlyComp_Unadj','CogTotalComp_Unadj','CogCrystalComp_Unadj' \
#     --output_root ./results_static_lr \
#     --dropout 0.1 \
#     --depth 2 \
#     --no_normalize_labels

# replace <combo_dir> with the printed "result combo_dir" path from run.py
# python compute_graphs.py --combo_dir <combo_dir>

# S1200： 'CogFluidComp_Unadj','CogEarlyComp_Unadj','CogTotalComp_Unadj','CogCrystalComp_Unadj'
#ABCD: 'nihtbx_fluidcomp_uncorrected','nihtbx_cryst_uncorrected','nihtbx_totalcomp_uncorrected'