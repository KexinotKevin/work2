#!/usr/bin/env bash
set -e

# 确保在根目录下运行


# 全局基础参数定义 (可以根据自己的需求调整 dataset_name, epoch 等)
BASE_ARGS="--use_dataset_cfg \
    --dataset_name S1200 \
    --atlas_name schaefer200_S1 \
    --sc_kinds FA fiber_count \
    --fc_kind pcc_rest \
    --label_types CogFluidComp_Unadj,CogEarlyComp_Unadj,CogTotalComp_Unadj,CogCrystalComp_Unadj \
    --num_epochs 100 \
    --depth 2 \
    --test_repeat 5 \
    --input_dimension 300 \
    --hidden_dimension 64 \
    --learning_rate 0.0005 \
    --dropout 0.5 \
    --l2_penalty 0.005 \
    --use_early_stopping"

echo "==========================================="
echo "       Starting Ablation Study"
echo "==========================================="
NOW=$(date +"%Y%m%d_%H%M%S")
BASE_OUT="/public/home/baitianyu/kexin/projects/work2/results/ablation"
# 1. 完整模型 (Ours)
echo ">>> Running 1/4: Full Model (Ours)"
python run.py $BASE_ARGS --output_root $BASE_OUT/$NOW/ours

# # 2. 移除对抗训练 (w/o GRL)
# echo ">>> Running 2/4: Model w/o GRL"
# python run.py $BASE_ARGS --output_root $BASE_OUT/$NOW/wo_grl --disable_grl

# # 3. 移除 PageRank 选择 (w/o PR)
# echo ">>> Running 3/4: Model w/o PageRank"
# python run.py $BASE_ARGS --output_root $BASE_OUT/$NOW/wo_pr --disable_pr

# # 4. 移除边特征学习 (w/o EL)
# echo ">>> Running 4/4: Model w/o Edge Learning"
# python run.py $BASE_ARGS --output_root $BASE_OUT/$NOW/wo_el --disable_el

# echo "==========================================="
# echo "   Training Complete. Generating Tables..."
# echo "==========================================="

# # 调用 Python 脚本生成 LaTeX 表格
# cd ablation
# python generate_table.py $NOW
