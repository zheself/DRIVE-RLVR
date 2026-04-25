# Stage2 消融实验设计与 r16 训练结果

## 一、r16 训练结果分析

### 1.1 训练概况

- 基础模型：stage1_v5_step135
- 数据：stage2_v2_str.jsonl（360 条 competition 难题）
- 配置：n_samples_per_prompt=16, rollout_batch_size=8, max_epochs=2
- 实际运行：45 global steps × 2 inner epochs = 90 训练迭代
- 训练时间：约 2.7 小时

### 1.2 训练指标

| 阶段 | Avg Score | Avg Reward | Avg Loss | Avg KL |
|------|-----------|------------|----------|--------|
| P1 (step 1-15) | 0.0785 | -0.0388 | 0.1351 | 0.000039 |
| P2 (step 16-30) | 0.0870 | -0.0178 | 0.1676 | 0.000235 |
| P3 (step 31-45) | 0.0878 | 0.0021 | 0.2788 | 0.000821 |

### 1.3 Checkpoint 对比

| Checkpoint | 窗口范围 | Avg Score | Max Score |
|------------|---------|-----------|-----------|
| step 20 | 15-25 | 0.0863 | 0.0914 |
| step 30 | 25-35 | 0.0874 | 0.0914 |
| step 40 | 35-45 | 0.0882 | 0.0914 |
| step 45 (final) | 40-45 | 0.0884 | 0.0906 |

**结论：final（step 45）即为最优，中间 checkpoint 无优势。**

### 1.4 关键发现

Competition 难题对 3B 模型来说太难了：
- Score 始终在 0.078-0.091 徘徊，提升幅度仅 1.2%
- Score 几乎全由格式分贡献（0.1 × 合规率），exec_score 接近 0
- 与 stage1 的显著提升（0.128→0.179）形成鲜明对比

---

## 二、消融实验设计修正

### 2.1 OpenRLHF 的 epoch 机制

OpenRLHF 中 `max_epochs` 控制的是每个 global step 内部的 **inner epoch** 数，而非外层数据遍历次数。例如 `max_epochs=2` 意味着每步的 experience buffer 会被训练 2 遍。

因此：
- r16：45 global steps × 2 inner epochs = 90 训练迭代
- 要让 r64 公平对比，也需要 45 global steps × 2 inner epochs = 90 训练迭代

### 2.2 修正后的实验设计

| | r16（对照组） | r64（实验组） |
|---|---|---|
| n_samples_per_prompt | 16 | 64 |
| rollout_batch_size | 8 | 2 |
| max_samples | 360 | 90 |
| max_epochs | 2 | 2 |
| global steps | 45 | 45 |
| inner epochs/step | 2 | 2 |
| 每步生成量 | 8×16=128 | 2×64=128 |
| 总生成量 | 5,760 | 5,760 |
| 总训练迭代 | 90 | 90 |
| 题目覆盖 | 360 题 | 90 题 |

### 2.3 核心 Tradeoff

唯一的自变量是 GRPO 组大小，带来的 tradeoff：

- **r16（广度优先）**：每步看 8 道不同的题，每题 16 个回答。覆盖全部 360 题，题目多样性高，但每题的组内对比信号较弱。
- **r64（深度优先）**：每步只看 2 道题，每题 64 个回答。只覆盖 90 题（1/4），但每题有 64 个样本做组内对比，advantage 估计更准确。对于难题（大部分回答得 0 分），64 个样本中更可能出现至少 1 个正确回答，从而产生有效的正向梯度信号。

### 2.4 实验假设

- 如果 r64 在只看 90 题的情况下，最终评估（全部 360 题）的通过率与 r16 持平或更高 → 大 rollout 的 GRPO 信号质量更重要
- 如果 r64 明显不如 r16 → 题目覆盖面更重要

---

## 三、缓存清理记录

| 清理项 | 大小 | 说明 |
|--------|------|------|
| r16 中间 checkpoint (step 20/30/40) | 105G | final 已保存，中间无优势 |
| 旧 ray sessions (2 个) | 730M | 训练已结束 |

---

## 四、文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `train_stage2_r64.sh` | 修改 | max_samples 180→90, max_epochs 1→2，与 r16 对齐 |
| `reports/10_v5训练结果分析报告.md` | 已有 | v5 完整训练结果 |
