#!/bin/bash
# train_stage2_r64.sh — Stage2 消融实验：Rollout=64
# 基于 stage1_v5_step135 继续训练，competition 难题 360 条
# 实验组：n_samples_per_prompt=64, rollout_batch_size=2
#
# 与 r16 的区别：
#   每步只看 2 道题但每题 64 个回答，GRPO 组内对比信号更强
#   固定 45 global steps × 2 inner epochs = 90 训练迭代（与 r16 完全一致）
#   通过 max_samples=90 控制步数：90/2=45 steps
#   预计 ~2.5 小时

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
    --save_path ./checkpoints/stage2_r64 \
    --input_key prompt \
    --label_key test_cases \
    --prompt_data /mnt/sdc/ubuntu/cjz_projects/DRIVE/data/stage2_v2_str.jsonl \
    --input_template "{}" \
    --max_samples 90 \
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
    --use_tensorboard ./checkpoints/stage2_r64 \
    --save_steps 30
