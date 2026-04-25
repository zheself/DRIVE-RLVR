# HumanEval 评测报告：四模型对比

## 一、评测设置

| 项目 | 说明 |
|------|------|
| 测试集 | OpenAI HumanEval（164 道函数补全题） |
| 数据来源 | `/mnt/sdc/ubuntu/cjz_projects/datasets/openai_humaneval` |
| 评测指标 | pass@1（greedy decoding，每题生成 1 次） |
| 生成参数 | greedy decoding, max_new_tokens=768 |
| 评分方式 | 拼接模型生成代码 + HumanEval 自带 check() 测试函数，执行 assert 判断通过与否 |
| Prompt 模板 | 与训练时一致：中文指令 + `<think>` 格式要求 + 函数签名 |

### 与 APPS 测试集的区别

| | APPS 测试集 | HumanEval |
|---|---|---|
| 题目类型 | 竞赛/面试题，stdin/stdout 或 fn_name 型 | 纯函数补全，给定签名和 docstring |
| 语言 | 中文 prompt 包装 | 英文 docstring |
| 评分方式 | 逐 test case 计算通过率（0~1 连续分） | 全部 assert 通过才算 pass（0/1 二值） |
| 与训练数据关系 | APPS 训练集的不重叠子集 | 完全独立的外部 benchmark |

HumanEval 是一个更严格的泛化测试——模型不仅要解决没见过的题目，还要适应完全不同的题目风格和语言。

## 二、总体结果

| 指标 | Baseline | Stage1 v5_step135 | Stage2 v2_r16 | Stage2 v2_r64 |
|------|----------|-------------------|---------------|---------------|
| **pass@1** | **0.7927 (130/164)** | **0.7927 (130/164)** | **0.7988 (131/164)** | **0.7988 (131/164)** |
| 格式合规率 | 13% (21/164) | 93% (152/164) | 97% (159/164) | **99% (162/164)** |
| 平均响应长度 | 771 chars | 754 chars | 722 chars | 711 chars |

Stage2 训练后 pass@1 微升 0.6pp（130→131），格式合规率进一步提升到 97-99%，响应更简洁。

## 三、Per-sample 对比分析

### 3.1 Baseline vs v5_step135

| | v5 通过 | v5 失败 | 合计 |
|---|---|---|---|
| Baseline 通过 | 113 | 17 | 130 |
| Baseline 失败 | 17 | 17 | 34 |
| 合计 | 130 | 34 | 164 |

- 两者都通过：113 题（68.9%）
- 仅 Baseline 通过（v5 回归）：17 题
- 仅 v5 通过（v5 新增）：17 题
- 两者都失败：17 题

### 3.2 v5_step135 vs Stage2 模型

| 对比 | 两者都通过 | 仅 v5 通过 | 仅 Stage2 通过 |
|------|-----------|-----------|---------------|
| v5 vs v2_r16 | 124 | 6 | 7 |
| v5 vs v2_r64 | 124 | 6 | 7 |

Stage2 训练后新解决了 7 题，丢失了 6 题，净增 1 题。

### 3.3 r16 vs r64

| | r64 通过 | r64 失败 | 合计 |
|---|---|---|---|
| r16 通过 | 127 | 4 | 131 |
| r16 失败 | 4 | 29 | 33 |
| 合计 | 131 | 33 | 164 |

两者总分完全一致（131/164），但 per-sample 有 4 题互换。

## 四、关键发现

### 4.1 pass@1 变化趋势：Baseline → Stage1 → Stage2

| 阶段 | pass@1 | 变化 |
|------|--------|------|
| Baseline | 79.27% (130) | — |
| Stage1 v5_step135 | 79.27% (130) | ±0 |
| Stage2 v2_r16 | 79.88% (131) | +0.6pp |
| Stage2 v2_r64 | 79.88% (131) | +0.6pp |

整个训练流程（Stage1 + Stage2）在 HumanEval 上仅微升 0.6pp，说明：

1. **无负迁移**：两阶段 GRPO 训练均未破坏模型在外部 benchmark 上的基础能力
2. **无显著正迁移**：APPS 上学到的代码能力没有大幅迁移到 HumanEval 风格的题目
3. **Stage2 competition 训练也是安全的**：没有因为在更难的数据上训练而损害简单题的能力

### 4.2 格式合规率持续提升

| 模型 | 格式合规率 |
|------|-----------|
| Baseline | 13% |
| v5_step135 | 93% |
| v2_r16 | 97% |
| v2_r64 | **99%** |

格式遵循能力随训练阶段持续提升。Stage2 训练进一步将合规率从 93% 推到 97-99%，几乎完美。r64（大 rollout）在格式合规上略优于 r16，可能因为更大的 GRPO 组提供了更强的格式对比信号。

### 4.3 消融对比：r16 vs r64 在 HumanEval 上无差异

两者 pass@1 完全一致（131/164），per-sample 仅 4 题互换。这与 competition 测试集上的结论一致——在 HumanEval 这种与训练数据差异较大的 benchmark 上，rollout 大小对最终能力没有显著影响。

### 4.4 与其他测试集结果的综合对比

| 测试集 | Baseline | v5_step135 | v2_r16 | v2_r64 |
|--------|----------|------------|--------|--------|
| APPS test_introductory (100 题) | 0.5447 | 0.5670 (+4.1%) | — | — |
| APPS test_competition (1000 题) | 0.1254 | 0.1467 (+17.0%) | 0.1516 (+3.3%) | 0.1457 (-0.7%) |
| HumanEval (164 题) | 0.7927 | 0.7927 (±0%) | 0.7988 (+0.8%) | 0.7988 (+0.8%) |

训练收益主要体现在同分布数据上（APPS），跨分布迁移（HumanEval）有限但无退化。

## 五、结论

1. **两阶段 GRPO 训练在 HumanEval 上安全无退化**：pass@1 从 79.27% 微升到 79.88%，无灾难性遗忘。

2. **格式遵循能力持续提升**：Baseline 13% → Stage1 93% → Stage2 97-99%，这是 GRPO 训练最稳定的跨域迁移效果。

3. **代码能力的跨域迁移有限**：APPS 上的训练收益没有大幅迁移到 HumanEval，但 79.88% 的 pass@1 对 3B 模型来说已接近天花板。

4. **消融对比**：r16 和 r64 在 HumanEval 上表现完全一致，rollout 大小对跨域泛化无影响。

## 六、文件清单

| 文件 | 说明 |
|------|------|
| `scripts/evaluate_humaneval.py` | HumanEval 评测脚本 |
| `eval_results/humaneval_baseline.json` | Baseline 评测结果 |
| `eval_results/humaneval_v5_step135.json` | v5_step135 评测结果 |
| `eval_results/humaneval_stage2_v2_r16.json` | Stage2 v2_r16 评测结果 |
| `eval_results/humaneval_stage2_v2_r64.json` | Stage2 v2_r64 评测结果 |
