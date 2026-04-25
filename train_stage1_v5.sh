#!/bin/bash
# train_stage1_v5.sh — GRPO 训练脚本 (OpenRLHF v0.10.1)
# v5 改进：
#   1. 数据量 1000→2847（全部 introductory + interview fn_name 型）
#   2. 修复 fn_name 误分类 bug（prepare_data 保留 fn_name 字段）
#   3. 降低学习率 1e-6→5e-7（防止 epoch 3 policy loss 飙升）
#   4. 增大 KL 系数 0.01→0.02（限制策略偏移）
#   5. max_new_tokens 768→1024（给复杂题更多空间）
#   6. save_steps 20→15（更密集保存，方便挑最优 checkpoint）

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
    --save_path ./checkpoints/stage1_v5 \
    --input_key prompt \
    --label_key test_cases \
    --prompt_data /mnt/sdc/ubuntu/cjz_projects/DRIVE/data/stage1_v2_str.jsonl \
    --input_template "{}" \
    --max_samples 2847 \
    --micro_train_batch_size 1 \
    --train_batch_size 128 \
    --rollout_batch_size 16 \
    --n_samples_per_prompt 8 \
    --max_new_tokens 768 \
    --max_epochs   1 \
    --zero_stage 3 \
    --gradient_checkpointing \
    --param_dtype bf16 \
    --advantage_estimator group_norm \
    --use_kl_loss \
    --init_kl_coef 0.02 \
    --temperature 1.0 \
    --actor_learning_rate 5e-7 \
    --remote_rm_url http://localhost:5000/scores \
    --stop_properly_penalty_coef 0.0 \
    --overlong_buffer_len 256 \
    --overlong_penalty_factor 1.0 \
    --use_tensorboard ./checkpoints/stage1_v5 \
    --save_steps 15
