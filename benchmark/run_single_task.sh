#!/bin/bash
# run_single_task.sh — 单独跑某个任务（调试用）
#
# 用法:
#   bash run_single_task.sh <model_path> <output_name> <task_name> [num_fewshot]
#
# 示例:
#   bash run_single_task.sh ../checkpoints/stage1_v5_step135 stage1_v5 humaneval 0
#   bash run_single_task.sh ../checkpoints/stage2_r64 stage2_r64 gsm8k 5

set -e

MODEL_PATH="${1:?用法: bash run_single_task.sh <model_path> <output_name> <task_name> [num_fewshot]}"
OUTPUT_NAME="${2:?用法: bash run_single_task.sh <model_path> <output_name> <task_name>}"
TASK_NAME="${3:?用法: bash run_single_task.sh <model_path> <output_name> <task_name>}"
NUM_FEWSHOT="${4:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/results/${OUTPUT_NAME}"
mkdir -p "${RESULT_DIR}"

FEWSHOT_ARG=""
if [ -n "${NUM_FEWSHOT}" ]; then
    FEWSHOT_ARG="--num_fewshot ${NUM_FEWSHOT}"
fi

echo "=== 单任务评测: ${TASK_NAME} ==="
echo "模型: ${MODEL_PATH}"
echo "输出: ${RESULT_DIR}"

lm_eval --model vllm \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096,tensor_parallel_size=1" \
    --tasks "${TASK_NAME}" \
    --batch_size auto \
    ${FEWSHOT_ARG} \
    --output_path "${RESULT_DIR}" \
    --log_samples \
    2>&1 | tee "${RESULT_DIR}/eval_${TASK_NAME}.log"

echo "=== 完成 ==="
