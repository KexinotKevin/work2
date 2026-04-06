#!/usr/bin/env bash
set -e

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate gnn_work2

export PYTHONPATH="$(pwd)/../../":$PYTHONPATH

# 生成共享时间戳，所有模型的结果保存到同一文件夹
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="../results/debug_baseline/${TIMESTAMP}"

# MODELS=("GCN" "GAT" "SAGE" "RelGNN" "BrainGNN" "BrainNetCNN" "GraphTransformer" "BNT")
# 加入新的三个模型
MODELS=("BrainRGIN")
MODALITIES=("SC" "FC" "SC_FC")
# MODALITIES=("SC" "FC")

for MODEL in "${MODELS[@]}"; do
    for MOD in "${MODALITIES[@]}"; do
        
        # RelGNN 强绑定 SC和FC的输入机制，单模态跳过
        if [ "$MODEL" == "RelGNN" ] && [ "$MOD" != "SC_FC" ]; then
            continue
        fi

        if [ "$MODEL" == "BNT" ] && [ "$MOD" == "SC_FC" ]; then
            continue
        fi

        echo ">>> Running Baseline | Model: ${MODEL} | Modality: ${MOD} | Timestamp: ${TIMESTAMP} <<<"
        
        python run_baseline.py \
            --model_type ${MODEL} \
            --modality ${MOD} \
            --use_dataset_cfg \
            --num_epochs 100 \
            --dataset_name S1200 \
            --atlas_name schaefer200_S1 \
            --num_nodes 216 \
            --sc_kinds FA \
            --fc_kind pcc_rest \
            --label_types 'CogFluidComp_Unadj','CogEarlyComp_Unadj','CogTotalComp_Unadj','CogCrystalComp_Unadj' \
            --output_root ${OUTPUT_ROOT} \
            --timestamp ${TIMESTAMP} \
            --disable_grl \
            --seed 42 \
            --use_early_stopping
    done
done
