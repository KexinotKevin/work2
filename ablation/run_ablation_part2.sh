#!/usr/bin/env bash
set -e

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
echo " Starting Ablation Part 2: Threshold & Pooling"
echo "==========================================="
NOW=$(date +"%Y%m%d_%H%M%S")
BASE_OUT="/public/home/baitianyu/kexin/projects/work2/results/ablation"
# ----- 实验组A：Consistency Threshold 影响 -----
# (注：0.75 为 Ours 默认设定，上一个脚本已跑过，这里补齐 0.25, 0.5, 1.0)
echo ">>> Running A-1: Consistency Threshold = 0.25"
python run.py $BASE_ARGS --cons_thresh 0.25 --output_root $BASE_OUT/$NOW/thresh_0.25

echo ">>> Running A-2: Consistency Threshold = 0.50"
python run.py $BASE_ARGS --cons_thresh 0.50 --output_root $BASE_OUT/$NOW/thresh_0.50

echo ">>> Running A-3: Consistency Threshold = 1.0"
python run.py $BASE_ARGS --cons_thresh 1.0 --output_root $BASE_OUT/$NOW/thresh_1.0

# ----- 实验组B：Pooling Strategy 影响 -----
# (注：concat 为 Ours 默认设定，上一个脚本已跑过，这里补齐 gmp 和 gap)
echo ">>> Running B-1: Pooling Strategy = GMP only"
python run.py $BASE_ARGS --pool_strategy gmp --output_root $BASE_OUT/$NOW/pool_gmp

echo ">>> Running B-2: Pooling Strategy = GAP only"
python run.py $BASE_ARGS --pool_strategy gap --output_root $BASE_OUT/$NOW/pool_gap

echo "==========================================="
echo "   Training Complete. Generating Tables..."
echo "==========================================="

cd /public/home/baitianyu/kexin/projects/work2/ablation
python generate_table_part2.py $NOW