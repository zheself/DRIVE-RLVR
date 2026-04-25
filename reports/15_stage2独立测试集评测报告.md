# Stage2 模型独立测试集评测报告

## 一、评测设置

| 项目 | 说明 |
|------|------|
| 测试集 | test_introductory（APPS test.jsonl，与训练集完全不重叠，全部 stdin 型） |
| 样本数 | 100 |
| 生成参数 | greedy decoding, max_new_tokens=768 |
| 评分 | format_score (0.1) + exec_score (0.9) |

Stage2 模型（r16、r64）是在 v5_step135 基础上用 competition 难题做 GRPO 训练的。本次评测检验：经过 competition 难题训练后，模型在 introductory 简单题上的能力是否受损。

## 二、全模型横向对比

| 模型 | Total | Format | Exec | Full Pass | Partial | Zero | 格式合规 | 平均长度 |
|------|-------|--------|------|-----------|---------|------|----------|----------|
| Baseline | 0.5447 | 0.0800 | 0.4647 | 32 | 35 | 33 | 80% | 932 |
| Stage1 v4 | 0.4768 | 0.0990 | 0.3778 | 28 | 25 | 47 | 99% | 853 |
| **Stage1 v5_step135** | **0.5670** | **0.0910** | **0.4760** | **35** | **27** | **38** | **91%** | **763** |
| Stage2 r16 | 0.5322 | 0.0940 | 0.4382 | 34 | 23 | 43 | 94% | 705 |
| Stage2 r64 | 0.5442 | 0.0960 | 0.4482 | 34 | 25 | 41 | 96% | 661 |

## 三、关键发现

### 3.1 Stage2 训练导致 introductory 能力轻微下降

| 对比 | Exec Score | 变化 | Full Pass | Zero Pass |
|------|-----------|------|-----------|-----------|
| v5_step135（stage2 基线） | 0.4760 | — | 35 | 38 |
| Stage2 r16 | 0.4382 | **-7.9%** | 34 (-1) | 43 (+5) |
| Stage2 r64 | 0.4482 | **-5.8%** | 34 (-1) | 41 (+3) |

两个 stage2 模型的 exec_score 都低于 v5_step135，说明 competition 难题的 GRPO 训练对 introductory 能力产生了轻微的负迁移。

r64 的下降幅度（-5.8%）小于 r16（-7.9%），与训练日志中 r64 的 policy loss 更稳定、KL 更可控的结论一致。

### 3.2 Per-sample 分析

**Stage2 r16 vs v5：**
- r16 更好：6 个
- r16 更差：12 个
- 持平：82 个
- v5 全通过但 r16 零分：3 个（严重回归）
- r16 全通过但 v5 零分：1 个

**Stage2 r64 vs v5：**
- r64 更好：8 个
- r64 更差：12 个
- 持平：80 个
- v5 全通过但 r64 零分：3 个（严重回归）
- r64 全通过但 v5 零分：1 个

两个 stage2 模型都呈现"更差的样本多于更好的样本"的模式（12 > 6, 12 > 8），且各有 3 个 v5 能全通过但 stage2 零分的严重回归。

### 3.3 r64 vs r16 直接对比

| | r64 更好 | r16 更好 | 持平 |
|---|---|---|---|
| 样本数 | 6 | 3 | 91 |

r64 在 introductory 上也略优于 r16，与训练阶段的消融结论一致：大 rollout 策略产生的模型更稳健。

### 3.4 格式合规率持续提升

| 模型 | 格式合规率 |
|------|-----------|
| Baseline | 80% |
| v5_step135 | 91% |
| Stage2 r16 | 94% |
| Stage2 r64 | **96%** |

Stage2 训练进一步提升了格式合规率，这是 GRPO 训练的一致性收益。

### 3.5 响应长度持续缩短

| 模型 | 平均长度 |
|------|----------|
| Baseline | 932 |
| v5_step135 | 763 |
| Stage2 r16 | 705 |
| Stage2 r64 | **661** |

从 baseline 到 stage2_r64，响应长度缩短了 29%。模型越来越倾向于生成简洁代码。

### 3.6 Exec Score 分布

| 区间 | Baseline | v5_step135 | Stage2 r16 | Stage2 r64 |
|------|----------|------------|------------|------------|
| 0（完全失败） | 33 | 38 | **43** | 41 |
| 0.01-0.25 | 12 | 8 | 5 | 6 |
| 0.26-0.50 | 14 | 9 | 13 | 12 |
| 0.51-0.75 | 6 | 7 | 3 | 5 |
| 0.76-0.99 | 3 | 3 | 2 | 2 |
| 1.0（全通过） | 32 | 35 | 34 | 34 |

Stage2 模型的 full pass 数量（34）接近 v5（35），但 zero pass 明显增加（43/41 vs 38）。这说明 stage2 训练没有损害模型解决"能解的题"的能力，但让一些"边缘题"从 partial pass 滑向了 zero pass。

## 四、完整训练链路总结

| 阶段 | 模型 | Exec Score | vs Baseline | vs 上一阶段 |
|------|------|-----------|-------------|-------------|
| — | Baseline | 0.4647 | — | — |
| Stage1 SFT (v4) | v4 | 0.3778 | **-18.7%** | — |
| Stage1 GRPO (v5) | v5_step135 | **0.4760** | **+2.4%** | — |
| Stage2 GRPO (r16) | r16 | 0.4382 | -5.7% | **-7.9%** |
| Stage2 GRPO (r64) | r64 | 0.4482 | -3.6% | **-5.8%** |

## 五、结论

1. **Stage2 的 competition 训练对 introductory 能力产生了负迁移**：r16 下降 7.9%，r64 下降 5.8%。这是预期内的——用难题训练可能导致模型在简单题上"过度思考"或改变解题策略。

2. **r64 的负迁移更小**：再次验证了大 rollout 策略的优势——更稳定的梯度信号不仅在 competition 上表现更好，对 introductory 能力的损害也更小。

3. **v5_step135 仍然是综合最优模型**：在 introductory 测试集上，v5 的 exec_score（0.476）高于所有其他模型。Stage2 训练在 competition 上的提升（从训练日志看 score 从 0.078 到 0.090）不足以弥补 introductory 上的损失。

4. **当前 stage2 的价值有限**：competition 题目对 3B 模型太难，训练信号稀疏，且会损害已有能力。建议考虑：
   - 用 interview 难度替代 competition 作为 stage2 数据（难度适中，信号更丰富）
   - 或在 stage2 中混入部分 introductory 数据做 replay，防止遗忘
   - 或直接跳过 stage2，专注优化 stage1 的训练策略

## 六、待完成：test_competition 评测

以上评测仅在 introductory 上进行，验证了 stage2 训练的负迁移。但 stage2 的本职目标是提升 competition 能力，需要在 test_competition 上评测才能完整评价。

### 待跑评测

| 模型 | 测试集 | 目的 |
|------|--------|------|
| v5_step135 | test_competition | 作为 stage2 在 competition 上的 baseline |
| stage2_r16 | test_competition | 评估 r16 在 competition 上的提升 |
| stage2_r64 | test_competition | 评估 r64 在 competition 上的提升 |

### 执行命令

```bash
tmux new-session -d -s eval_comp -c /mnt/sdc/ubuntu/cjz_projects/DRIVE \
  'conda run -n cjz_openrlhf python3 scripts/evaluate_offline.py \
    --model_path checkpoints/stage1_v5_step135 \
    --test_data data/test_competition.jsonl \
    --output eval_results/v5_step135_test_comp.json \
    --max_new_tokens 768 2>&1 | tee /tmp/eval_comp.log && \
  conda run -n cjz_openrlhf python3 scripts/evaluate_offline.py \
    --model_path checkpoints/stage2_r16 \
    --test_data data/test_competition.jsonl \
    --output eval_results/stage2_r16_test_comp.json \
    --max_new_tokens 768 2>&1 | tee -a /tmp/eval_comp.log && \
  conda run -n cjz_openrlhf python3 scripts/evaluate_offline.py \
    --model_path checkpoints/stage2_r64 \
    --test_data data/test_competition.jsonl \
    --output eval_results/stage2_r64_test_comp.json \
    --max_new_tokens 768 2>&1 | tee -a /tmp/eval_comp.log && \
  echo "ALL DONE" >> /tmp/eval_comp.log'
```

### 预期

根据训练日志，competition 上 exec_score 接近 0（score 几乎全由格式分贡献），三个模型在 test_competition 上的 exec_score 可能都很低，区分度有限。但仍有必要跑一次以获得完整数据。
