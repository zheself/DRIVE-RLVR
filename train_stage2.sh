#!/bin/bash
# train_stage2.sh — Stage2 GRPO 训练脚本 (OpenRLHF v0.10.1)
# 基于 stage1_v5 最优 checkpoint (step 135) 继续训练
# 数据：competition 难度 360 条（全部 stdin 型，已修复 list input 归一化）
#
# 与 stage1 的关键差异：
#   - pretrain 使用 stage1_v5_step135（而非原始模型）
#   - 数据量小（360 vs 2847），需要多 epoch 但防过拟合
#   - lr 进一步降低到 3e-7（在已训练模型上微调，避免灾难性遗忘）
#   - KL 系数提高到 0.03（competition 题更难，模型容易走偏）
#   - 3 epochs，共 ~69 步，约 3 小时

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
    --save_path ./checkpoints/stage2_v1 \
    --input_key prompt \
    --label_key test_cases \
    --prompt_data /mnt/sdc/ubuntu/cjz_projects/DRIVE/data/stage2_v2_str.jsonl \
    --input_template "{}" \
    --max_samples 360 \
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
    --init_kl_coef 0.03 \
    --temperature 1.0 \
    --actor_learning_rate 3e-7 \
    --remote_rm_url http://localhost:5000/scores \
    --stop_properly_penalty_coef 0.0 \
    --overlong_buffer_len 256 \
    --overlong_penalty_factor 1.0 \
    --use_tensorboard ./checkpoints/stage2_v1 \
    --save_steps 10
