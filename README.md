[**中文版**](README_zh.md) | English

# DRIVE-RLVR

**Two-stage GRPO training for code generation with verifiable rewards**

A reinforcement learning project that improves the code generation ability of Qwen2.5-Coder-3B-Instruct using GRPO (Group Relative Policy Optimization) with execution-based verifiable rewards (RLVR). Built on the OpenRLHF framework with a custom reward service that evaluates generated code by actually running it against test cases.

## Highlights

- **Pure GRPO training** without Critic network — 50% less GPU memory than PPO
- **Verifiable rewards (RLVR)** — code correctness judged by real execution, not a learned reward model
- **Two-stage curriculum** — easy problems first (Stage1: 2847 introductory/interview), then hard (Stage2: 302 competition)
- **Rollout size ablation** — n_samples_per_prompt=16 vs 64, studying GRPO group size effects on sparse-reward tasks
- **HumanEval pass@1 = 79.88%** with zero degradation from RL training

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    OpenRLHF (Ray + vLLM)                │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐    │
│  │  Actor    │  │   Ref    │  │   vLLM Engine      │    │
│  │  (3B)    │◄─┤  (3B)    │  │   (Generation)     │    │
│  │ GPU 0,1  │  │ colocate │  │   GPU 2            │    │
│  └────┬─────┘  └──────────┘  └────────────────────┘    │
│       │                                                 │
│       │  KL regularization                              │
│       ▼                                                 │
│  ┌──────────────────────┐    ┌──────────────────────┐   │
│  │  GRPO Advantage      │    │  Reward Service      │   │
│  │  A_i = (r_i - μ) / σ │◄───┤  (Flask, CPU)        │   │
│  │  (group_norm)        │    │  Code Execution      │   │
│  └──────────────────────┘    │  + Format Check      │   │
│                              └──────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Training Pipeline

### Stage 1: Foundation (introductory + interview)

| Config | Value |
|--------|-------|
| Base model | Qwen2.5-Coder-3B-Instruct |
| Data | 2847 problems (APPS introductory + interview fn_name) |
| Algorithm | GRPO (advantage_estimator=group_norm) |
| Learning rate | 5e-7 |
| KL coefficient | 0.02 |
| n_samples_per_prompt | 8 |
| Steps | 177 (1 epoch) |
| Hardware | 4× RTX A6000 (48GB), ZeRO-3 |

### Stage 2: Advanced (competition) + Ablation

| Config | r16 (control) | r64 (experiment) |
|--------|---------------|-------------------|
| Base model | Stage1 v5 step135 | Stage1 v5 step135 |
| Data | 302 competition (cleaned) | 302 competition (cleaned) |
| n_samples_per_prompt | 16 | 64 |
| rollout_batch_size | 8 | 2 |
| Problems per step | 8 | 2 |
| Samples per problem | 16 | 64 |
| Total generations | 4,736 | 4,736 |

## Results

### Independent Test Set (APPS test_introductory, 100 problems)

| Model | Total Score | Exec Score | Format Rate | Full Pass |
|-------|------------|------------|-------------|-----------|
| Baseline | 0.5447 | 0.4647 | 80% | 32 |
| Stage1 v5_step135 | **0.5670** | **0.4760** | 91% | **35** |
| Stage2 r16 | 0.5322 | 0.4382 | 94% | 34 |
| Stage2 r64 | 0.5442 | 0.4482 | 96% | 34 |

### HumanEval (164 problems, external benchmark)

| Model | pass@1 | Format Rate |
|-------|--------|-------------|
| Baseline | 79.27% (130/164) | 13% |
| Stage1 v5_step135 | 79.27% (130/164) | 93% |
| Stage2 v2_r16 | **79.88% (131/164)** | 97% |
| Stage2 v2_r64 | **79.88% (131/164)** | 99% |

### APPS test_competition (1000 problems)

| Model | Exec Score | Format Rate | Zero Pass |
|-------|-----------|-------------|-----------|
| Baseline | 0.1254 | 67% | 733 |
| v5_step135 | 0.1467 | 43% | 684 |
| Stage2 v2_r16 | **0.1516** | 63% | **671** |
| Stage2 v2_r64 | 0.1457 | 66% | 683 |

## Reward Design

The reward function combines execution correctness with format compliance:

```
R = exec_score + format_score

exec_score = passed_test_cases / total_test_cases    ∈ [0, 1.0]
format_score = 0.1 if (<think> tag AND ```python block)  ∈ {0, 0.1}
```

Code execution runs in a sandboxed subprocess with:
- 5-second timeout (kills infinite loops)
- 512MB memory limit (prevents memory bombs)
- Process isolation (crashes don't affect the reward service)
- Float-tolerant matching (1e-4 tolerance for numerical outputs)

## Key Findings

1. **Reward quality is everything** — A buggy reward function caused 3 rounds of wasted training. The model optimized the bug perfectly (format-only hacking, exec_score=0).

2. **Large rollout helps on hard problems** — r64 showed 2.2× faster score improvement than r16 on competition problems, because 64 samples are more likely to contain at least one correct solution for GRPO to learn from.

3. **KL coefficient matters** — Too small (0.01): overfitting (train +69%, test -18.7%). Optimal (0.02): stable training with real generalization.

4. **Curriculum learning works** — Starting with easy problems ensures dense reward signals. Jumping to competition directly yields near-zero exec_score.

5. **3B model ceiling** — introductory exec≈0.48, competition exec≈0.15, HumanEval pass@1≈79.9%. Competition problems (graph theory, advanced DP) are fundamentally hard for 3B.

## Project Structure

```
DRIVE/
├── reward_service.py        # Flask reward server (OpenRLHF remote_rm_url)
├── reward_verifier.py       # Core: code execution + format checking
├── prepare_data.py          # APPS dataset preprocessing & splitting
├── train_stage1_v5.sh       # Stage1 GRPO training script
├── train_stage2_v2_r16.sh   # Stage2 ablation: small rollout
├── train_stage2_v2_r64.sh   # Stage2 ablation: large rollout
├── scripts/
│   ├── evaluate_offline.py  # Offline evaluation on APPS
│   ├── evaluate_humaneval.py# HumanEval benchmark evaluation
│   ├── clean_competition.py # Data cleaning with reference solutions
│   └── compare_results.py   # Cross-model comparison reports
├── data/                    # Processed training & test datasets
├── eval_results/            # Evaluation result JSONs
├── reports/                 # 19 detailed development reports (Chinese)
└── checkpoints/             # Model checkpoints (not in repo)
```

## Quick Start

### 1. Start the reward service

```bash
python reward_service.py
# Listening on http://0.0.0.0:5000
```

### 2. Run Stage1 training

```bash
bash train_stage1_v5.sh
```

### 3. Run Stage2 ablation

```bash
# Control group (n=16)
bash train_stage2_v2_r16.sh

# Experiment group (n=64)
bash train_stage2_v2_r64.sh
```

### 4. Evaluate

```bash
python scripts/evaluate_offline.py \
    --model_path checkpoints/stage1_v5_step135 \
    --test_data data/test_introductory.jsonl \
    --output eval_results/v5_test_intro.json

python scripts/evaluate_humaneval.py \
    --model_path checkpoints/stage1_v5_step135 \
    --output eval_results/humaneval_v5.json
```

## Requirements

- 4× NVIDIA GPU with ≥48GB VRAM (tested on RTX A6000)
- OpenRLHF v0.10.1
- vLLM, Ray, DeepSpeed (ZeRO-3)
- Qwen2.5-Coder-3B-Instruct base model
- APPS dataset (codeparrot/apps)

## Development Reports

The `reports/` directory contains 19 detailed reports documenting the entire development process, including problem diagnosis, training analysis, bug fixes, and ablation studies. These reports are in Chinese.

## License

MIT
