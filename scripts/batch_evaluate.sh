#!/bin/bash
# 批量评估脚本（离线版，不依赖 reward service）

set -e
cd /mnt/sdc/ubuntu/cjz_projects/DRIVE

TEST_DATA="data/stage1_base_str.jsonl"
OUTPUT_DIR="evaluation_results"
MAX_SAMPLES=100
MAX_NEW_TOKENS=768

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "DRIVE Stage1 模型评估（离线版）"
echo "=========================================="

source ~/anaconda3/etc/profile.d/conda.sh
conda activate cjz_openrlhf

# 模型列表（按顺序评估）
declare -a MODEL_NAMES=("v2" "v3" "baseline")
declare -A MODEL_PATHS=(
    ["baseline"]="/mnt/sdc/ubuntu/cjz_projects/models/Qwen2.5-Coder-3B-Instruct"
    ["v2"]="/mnt/sdc/ubuntu/cjz_projects/DRIVE/checkpoints/stage1_v2"
    ["v3"]="/mnt/sdc/ubuntu/cjz_projects/DRIVE/checkpoints/stage1_v3"
)

for model_name in "${MODEL_NAMES[@]}"; do
    model_path="${MODEL_PATHS[$model_name]}"
    output_file="$OUTPUT_DIR/${model_name}_results.json"

    echo ""
    echo "=========================================="
    echo "评估模型: $model_name"
    echo "路径: $model_path"
    echo "=========================================="

    if [ -f "$output_file" ]; then
        echo "结果已存在，跳过。删除 $output_file 可重新评估。"
        continue
    fi

    python scripts/evaluate_offline.py \
        --model_path "$model_path" \
        --test_data "$TEST_DATA" \
        --output "$output_file" \
        --max_samples "$MAX_SAMPLES" \
        --max_new_tokens "$MAX_NEW_TOKENS"
done

echo ""
echo "=========================================="
echo "全部评估完成！结果在 $OUTPUT_DIR/"
echo "=========================================="
