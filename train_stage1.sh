#!/bin/bash
# train_stage1.sh — GRPO 训练脚本 (OpenRLHF v0.10.1)
# 使用 train_ppo_ray + advantage_estimator=group_norm 实现 GRPO  
# v4: 使用修复后的 reward_service，从原始模型开始训练
# 配置沿用 v2 的成功参数（1 epoch, lr=1e-6, temperature=1.0）

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
    --pretrain /mnt/sdc/ubuntu/cjz_projects/models/Qwen2.5-Coder-3B-Instruct \
    --save_path ./checkpoints/stage1_v4 \
    --input_key prompt \
    --label_key test_cases \
    --prompt_data /mnt/sdc/ubuntu/cjz_projects/DRIVE/data/stage1_base_str.jsonl \
    --input_template "{}" \
    --max_samples 1000 \
    --micro_train_batch_size 1 \
    --train_batch_size 128 \
    --rollout_batch_size 16 \
    --n_samples_per_prompt 8 \
    --max_new_tokens 768 \
    --max_epochs 3 \
    --zero_stage 3 \
    --gradient_checkpointing \
    --param_dtype bf16 \
    --advantage_estimator group_norm \
    --use_kl_loss \
    --init_kl_coef 0.01 \
    --temperature 1.0 \
    --actor_learning_rate 1e-6 \
    --remote_rm_url http://localhost:5000/scores \
    --stop_properly_penalty_coef 0.0 \
    --overlong_buffer_len 256 \
    --overlong_penalty_factor 1.0 \
    --use_tensorboard ./checkpoints/stage1_v4 \
    --save_steps 20
