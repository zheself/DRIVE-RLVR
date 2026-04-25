#!/bin/bash
# run_benchmark.sh — 使用 lm_eval 对指定模型跑通用 benchmark
#
# 用法:
#   bash run_benchmark.sh <model_path> <output_name> [task_group]
#
# 示例:
#   bash run_benchmark.sh /mnt/sdc/ubuntu/cjz_projects/models/Qwen2.5-Coder-3B-Instruct baseline
#   bash run_benchmark.sh ../checkpoints/stage1_v5_step135 stage1_v5 code
#   bash run_benchmark.sh ../checkpoints/stage2_r64 stage2_r64 all
#
# task_group 可选值:
#   all      — 全部任务（默认）
#   code     — 仅代码任务 (HumanEval, MBPP)
#   reason   — 仅推理任务 (GSM8K, ARC, HellaSwag, WinoGrande)
#   knowledge — 仅知识任务 (MMLU, TruthfulQA)

set -e

MODEL_PATH="${1:?用法: bash run_benchmark.sh <model_path> <output_name> [task_group] [gpu_id]}"
OUTPUT_NAME="${2:?用法: bash run_benchmark.sh <model_path> <output_name> [task_group] [gpu_id]}"
TASK_GROUP="${3:-all}"
GPU_ID="${4:-0}"

# 强制设置 GPU 绑定（覆盖外层设置，确保 vllm 子进程也继承）
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# 使用 HuggingFace 镜像（国内网络无法直接访问 huggingface.co）
export HF_ENDPOINT="https://hf-mirror.com"
# 允许 code_eval 执行模型生成的代码（HumanEval/MBPP 需要）
export HF_ALLOW_CODE_EVAL="1"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/results/${OUTPUT_NAME}"
mkdir -p "${RESULT_DIR}"

# ============================================================
# 任务定义
# ============================================================
# 代码生成
CODE_TASKS="humaneval,mbpp"
# 数学推理
MATH_TASKS="gsm8k"
# 常识推理
COMMONSENSE_TASKS="hellaswag,arc_challenge,winogrande"
# 通用知识
KNOWLEDGE_TASKS="mmlu,truthfulqa_mc2"

case "${TASK_GROUP}" in
    code)      TASKS="${CODE_TASKS}" ;;
    reason)    TASKS="${MATH_TASKS},${COMMONSENSE_TASKS}" ;;
    knowledge) TASKS="${KNOWLEDGE_TASKS}" ;;
    all)       TASKS="${CODE_TASKS},${MATH_TASKS},${COMMONSENSE_TASKS},${KNOWLEDGE_TASKS}" ;;
    *)         echo "未知 task_group: ${TASK_GROUP}"; exit 1 ;;
esac

echo "============================================"
echo "  lm_eval Benchmark"
echo "============================================"
echo "  模型:     ${MODEL_PATH}"
echo "  输出名:   ${OUTPUT_NAME}"
echo "  任务组:   ${TASK_GROUP}"
echo "  GPU:      ${GPU_ID}"
echo "  任务列表: ${TASKS}"
echo "  结果目录: ${RESULT_DIR}"
echo "============================================"

# ============================================================
# 运行评测
# ============================================================
# 使用 vllm 后端加速推理，单卡即可跑 3B 模型
# --batch_size auto 让 lm_eval 自动选择最优 batch size
# --num_fewshot 根据任务类型设置（代码生成 0-shot，其他按标准设置）

lm_eval --model vllm \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,gpu_memory_utilization=0.5,max_model_len=4096,tensor_parallel_size=1" \
    --tasks "${TASKS}" \
    --batch_size auto \
    --output_path "${RESULT_DIR}" \
    --log_samples \
    --confirm_run_unsafe_code \
    2>&1 | tee "${RESULT_DIR}/eval.log"

echo ""
echo "=== 评测完成 ==="
echo "结果保存在: ${RESULT_DIR}"
