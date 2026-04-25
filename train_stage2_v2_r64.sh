#!/bin/bash
# train_stage2_v2_r64.sh — Stage2 v2 消融实验：大 Rollout (r64)
# 基于 stage1_v5_step135 继续训练
# 数据：stage2_v3（清洗后 302 条 competition，修复浮点精度 + 过滤噪声标注）
# 评分：reward_service.py（已修复浮点容差 1e-4，内存 512MB，超时 5s）
#
# 消融设计（与 r16 对齐）：
#   max_samples=74, rollout_batch_size=2 → 37 global steps
#   n_samples_per_prompt=64 → 每步生成 2×64=128
#   max_epochs=2 → 37 steps × 2 inner epochs = 74 训练迭代
#   总生成量 = 37×128 = 4,736（与 r16 一致）
#
# 注意：启动前需先运行 reward_service.py（不需要 GPU，纯 CPU 执行）
#   python reward_service.py
# 训练使用全部 4 张 GPU（ref 2卡 + actor 2卡，colocate 共享）

set -e

python -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --colocate_actor_ref \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 0 \
    --critic_num_nodes 0 \
    --critic_num_gpus_per_node 0 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.8 \
    --pretrain ./checkpoints/stage1_v5_step135 \
    --save_path ./checkpoints/stage2_v2_r64 \
    --input_key prompt \
    --label_key test_cases \
    --prompt_data /mnt/sdc/ubuntu/cjz_projects/DRIVE/data/stage2_v3_str.jsonl \
    --input_template "{}" \
    --max_samples 74 \
    --micro_train_batch_size 1 \
    --train_batch_size 128 \
    --rollout_batch_size 2 \
    --n_samples_per_prompt 64 \
    --max_new_tokens 768 \
    --max_epochs 2 \
    --zero_stage 3 \
    --gradient_checkpointing \
    --param_dtype bf16 \
    --advantage_estimator group_norm \
    --use_kl_loss \
    --init_kl_coef 0.03 \
    --temperature 1.0 \
    --actor_learning_rate 3e-7 \
    --remote_rm_url http://localhost:5000/scores \
    --stop_properly_penalty_coef 0.0 \
    --overlong_buffer_len 256 \
    --overlong_penalty_factor 1.0 \
    --use_tensorboard ./checkpoints/stage2_v2_r64 \
    --save_steps 10
