#!/bin/bash
# install_lm_eval.sh — 安装 lm-evaluation-harness
# 在 cjz_openrlhf conda 环境中安装

set -e

echo "=== 安装 lm-evaluation-harness ==="

# 激活环境（如果在 conda 外运行）
# conda activate cjz_openrlhf

pip install lm-eval[vllm]==0.4.8

echo "=== 验证安装 ==="
python -c "import lm_eval; print(f'lm_eval version: {lm_eval.__version__}')"

echo "=== 列出可用任务 ==="
lm_eval --tasks list 2>/dev/null | head -30

echo "=== 安装完成 ==="
