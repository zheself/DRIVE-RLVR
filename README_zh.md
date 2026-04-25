中文 | [**English**](README.md)

# DRIVE-RLVR

**基于 GRPO 算法的代码生成强化学习训练，使用可验证执行奖励**

本项目使用 GRPO（Group Relative Policy Optimization）算法对 Qwen2.5-Coder-3B-Instruct 进行代码生成能力的强化学习训练。采用 RLVR（Reinforcement Learning with Verifiable Rewards）范式——通过实际执行生成的代码并与测试用例比对来计算奖励信号，而非训练一个 reward model。基于 OpenRLHF 框架构建，包含自定义的代码执行奖励服务。

## 项目亮点

- **纯 GRPO 训练**，无需 Critic 网络——相比 PPO 节省 50% GPU 显存
- **可验证奖励（RLVR）**——代码正确性由真实执行结果判定，非学习的 reward model
- **两阶段课程学习**——先简单题（Stage1: 2847 道 introductory/interview），再难题（Stage2: 302 道 competition）
- **Rollout 大小消融实验**——n_samples_per_prompt=16 vs 64，研究 GRPO 组大小对稀疏奖励任务的影响
- **HumanEval pass@1 = 79.88%**，RL 训练零退化

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    OpenRLHF (Ray + vLLM)                │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐    │
│  │  Actor    │  │   Ref    │  │   vLLM Engine      │    │
│  │  (3B)    │◄─┤  (3B)    │  │   (生成阶段)        │    │
│  │ GPU 0,1  │  │ colocate │  │   GPU 2            │    │
│  └────┬─────┘  └──────────┘  └────────────────────┘    │
│       │                                                 │
│       │  KL 正则化                                       │
│       ▼                                                 │
│  ┌──────────────────────┐    ┌──────────────────────┐   │
│  │  GRPO Advantage      │    │  Reward Service      │   │
│  │  A_i = (r_i - μ) / σ │◄───┤  (Flask, CPU)        │   │
│  │  (group_norm)        │    │  代码执行 + 格式检查    │   │
│  └──────────────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 训练流程

### Stage 1：基础能力训练（introductory + interview）

| 配置 | 值 |
|------|-----|
| 基座模型 | Qwen2.5-Coder-3B-Instruct |
| 训练数据 | 2847 道题（APPS introductory + interview fn_name 型） |
| 算法 | GRPO (advantage_estimator=group_norm) |
| 学习率 | 5e-7 |
| KL 系数 | 0.02 |
| n_samples_per_prompt | 8 |
| 训练步数 | 177 步（1 epoch） |
| 硬件 | 4× RTX A6000 (48GB)，ZeRO-3 |

### Stage 2：进阶训练（competition）+ 消融实验

| 配置 | r16（对照组） | r64（实验组） |
|------|-------------|-------------|
| 基座模型 | Stage1 v5 step135 | Stage1 v5 step135 |
| 训练数据 | 302 道 competition（清洗后） | 302 道 competition（清洗后） |
| n_samples_per_prompt | 16 | 64 |
| rollout_batch_size | 8 | 2 |
| 每步题目数 | 8 道 | 2 道 |
| 每题采样数 | 16 个 | 64 个 |
| 总生成量 | 4,736 | 4,736 |

## 实验结果

### 独立测试集（APPS test_introductory，100 道题）

| 模型 | 总分 | 执行分 | 格式合规率 | 完全通过 |
|------|------|--------|-----------|---------|
| Baseline | 0.5447 | 0.4647 | 80% | 32 |
| Stage1 v5_step135 | **0.5670** | **0.4760** | 91% | **35** |
| Stage2 r16 | 0.5322 | 0.4382 | 94% | 34 |
| Stage2 r64 | 0.5442 | 0.4482 | 96% | 34 |

### HumanEval（164 道题，外部 benchmark）

| 模型 | pass@1 | 格式合规率 |
|------|--------|-----------|
| Baseline | 79.27% (130/164) | 13% |
| Stage1 v5_step135 | 79.27% (130/164) | 93% |
| Stage2 v2_r16 | **79.88% (131/164)** | 97% |
| Stage2 v2_r64 | **79.88% (131/164)** | 99% |

### APPS test_competition（1000 道题）

| 模型 | 执行分 | 格式合规率 | 零分题数 |
|------|--------|-----------|---------|
| Baseline | 0.1254 | 67% | 733 |
| v5_step135 | 0.1467 | 43% | 684 |
| Stage2 v2_r16 | **0.1516** | 63% | **671** |
| Stage2 v2_r64 | 0.1457 | 66% | 683 |

## 奖励函数设计

奖励函数结合代码执行正确性与格式合规性：

```
R = exec_score + format_score

exec_score = 通过的测试用例数 / 总测试用例数    ∈ [0, 1.0]
format_score = 0.1（当包含 <think> 标签和 ```python 代码块时）  ∈ {0, 0.1}
```

代码执行在沙箱化的子进程中运行，具备三层防护：
- 5 秒超时（终止死循环）
- 512MB 内存限制（防止内存炸弹）
- 进程隔离（崩溃不影响奖励服务）
- 浮点容差匹配（1e-4 容差，处理数值输出精度问题）

## 核心发现

1. **Reward 质量决定一切** — 有 bug 的 reward 函数导致 3 轮训练白费。模型完美地优化了 bug 本身（只学格式，exec_score=0）。

2. **大 rollout 在难题上更有效** — r64 的 score 提升速度是 r16 的 2.2 倍。因为 64 个采样中更可能包含至少 1 个正确解，为 GRPO 提供有效的正向梯度信号。

3. **KL 系数至关重要** — 太小（0.01）：过拟合（训练集 +69%，测试集 -18.7%）。最优值（0.02）：稳定训练，真实泛化。

4. **课程学习有效** — 先用简单题保证 reward 信号密度。直接用 competition 题会导致 exec_score≈0，GRPO 退化为纯格式优化。

5. **3B 模型能力天花板** — introductory exec≈0.48，competition exec≈0.15，HumanEval pass@1≈79.9%。competition 题目（图论、高级 DP）对 3B 模型来说根本性地困难。

## 项目结构

```
DRIVE/
├── reward_service.py        # Flask 奖励服务（OpenRLHF remote_rm_url 端点）
├── reward_verifier.py       # 核心：代码执行 + 格式检查
├── prepare_data.py          # APPS 数据集预处理与分割
├── train_stage1_v5.sh       # Stage1 GRPO 训练脚本
├── train_stage2_v2_r16.sh   # Stage2 消融实验：小 rollout
├── train_stage2_v2_r64.sh   # Stage2 消融实验：大 rollout
├── scripts/
│   ├── evaluate_offline.py  # APPS 离线评测
│   ├── evaluate_humaneval.py# HumanEval benchmark 评测
│   ├── clean_competition.py # 用参考解法清洗数据
│   └── compare_results.py   # 跨模型对比报告生成
├── data/                    # 处理后的训练和测试数据集
├── eval_results/            # 评测结果 JSON 文件
├── reports/                 # 19 份详细开发报告
└── checkpoints/             # 模型 checkpoint（不含在仓库中）
```

## 快速开始

### 1. 启动奖励服务

```bash
python reward_service.py
# 监听 http://0.0.0.0:5000
```

### 2. 运行 Stage1 训练

```bash
bash train_stage1_v5.sh
```

### 3. 运行 Stage2 消融实验

```bash
# 对照组 (n=16)
bash train_stage2_v2_r16.sh

# 实验组 (n=64)
bash train_stage2_v2_r64.sh
```

### 4. 评测

```bash
python scripts/evaluate_offline.py \
    --model_path checkpoints/stage1_v5_step135 \
    --test_data data/test_introductory.jsonl \
    --output eval_results/v5_test_intro.json

python scripts/evaluate_humaneval.py \
    --model_path checkpoints/stage1_v5_step135 \
    --output eval_results/humaneval_v5.json
```

## 环境要求

- 4× NVIDIA GPU，单卡显存 ≥48GB（测试环境：RTX A6000）
- OpenRLHF v0.10.1
- vLLM、Ray、DeepSpeed (ZeRO-3)
- Qwen2.5-Coder-3B-Instruct 基座模型
- APPS 数据集 (codeparrot/apps)

## 开发报告

`reports/` 目录包含 19 份详细的开发报告，记录了完整的开发过程，包括问题诊断、训练分析、bug 修复和消融实验。

## License

MIT
