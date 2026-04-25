#!/bin/bash
# run_all_models.sh — 三个模型并行跑 benchmark（各占一张 GPU）
#
# GPU 分配:
#   GPU 1 — Baseline (Qwen2.5-Coder-3B-Instruct)（避开 GPU 0 上的其他进程）
#   GPU 2 — Stage1 v5_step135
#   GPU 3 — Stage2 r64
#
# 用法:
#   bash run_all_models.sh [task_group]
#   task_group: all(默认), code, reason, knowledge

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_GROUP="${1:-all}"

BASELINE="/mnt/sdc/ubuntu/cjz_projects/models/Qwen2.5-Coder-3B-Instruct"
STAGE1="/mnt/sdc/ubuntu/cjz_projects/DRIVE/checkpoints/stage1_v5_step135"
STAGE2="/mnt/sdc/ubuntu/cjz_projects/DRIVE/checkpoints/stage2_r64"

LOG_DIR="${SCRIPT_DIR}/results"
mkdir -p "${LOG_DIR}"

echo "============================================"
echo "  并行 Benchmark 评测（3 模型 × 3 GPU）"
echo "  任务组: ${TASK_GROUP}"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""
echo "  GPU 1 → Baseline"
echo "  GPU 2 → Stage1 v5_step135"
echo "  GPU 3 → Stage2 r64"
echo ""

# 并行启动三个评测，各绑定不同 GPU
# 使用 env CUDA_VISIBLE_DEVICES 确保 vllm 子进程也继承 GPU 绑定
CUDA_VISIBLE_DEVICES=1 bash "${SCRIPT_DIR}/run_benchmark.sh" "${BASELINE}" "baseline"   "${TASK_GROUP}" 1 \
    2>&1 | tee "${LOG_DIR}/baseline.log" &
PID_BASELINE=$!

CUDA_VISIBLE_DEVICES=2 bash "${SCRIPT_DIR}/run_benchmark.sh" "${STAGE1}"   "stage1_v5"  "${TASK_GROUP}" 2 \
    2>&1 | tee "${LOG_DIR}/stage1_v5.log" &
PID_STAGE1=$!

CUDA_VISIBLE_DEVICES=3 bash "${SCRIPT_DIR}/run_benchmark.sh" "${STAGE2}"   "stage2_r64" "${TASK_GROUP}" 3 \
    2>&1 | tee "${LOG_DIR}/stage2_r64.log" &
PID_STAGE2=$!

echo "已启动 3 个并行评测进程:"
echo "  Baseline   PID=${PID_BASELINE}"
echo "  Stage1 v5  PID=${PID_STAGE1}"
echo "  Stage2 r64 PID=${PID_STAGE2}"
echo ""
echo "等待全部完成..."

# 等待所有进程，记录各自退出状态
FAIL=0

wait ${PID_BASELINE} || { echo "[FAIL] Baseline 评测失败"; FAIL=1; }
echo "[$(date '+%H:%M:%S')] Baseline 完成"

wait ${PID_STAGE1} || { echo "[FAIL] Stage1 v5 评测失败"; FAIL=1; }
echo "[$(date '+%H:%M:%S')] Stage1 v5 完成"

wait ${PID_STAGE2} || { echo "[FAIL] Stage2 r64 评测失败"; FAIL=1; }
echo "[$(date '+%H:%M:%S')] Stage2 r64 完成"

echo ""
echo "============================================"
echo "  全部评测完成"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
if [ ${FAIL} -ne 0 ]; then
    echo "  [警告] 有评测失败，请检查日志"
fi
echo "============================================"
echo ""
echo "生成对比报告:"
echo "  python ${SCRIPT_DIR}/compare_results.py"

exit ${FAIL}
