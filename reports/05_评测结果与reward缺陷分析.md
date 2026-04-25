# DRIVE Stage1 评测报告

**日期**：2026-04-19

---

## 一、评测概述

对三个模型在训练集前 100 条样本上进行离线评测（贪心解码，max_new_tokens=768）。

评分逻辑与训练时一致（`reward_verifier.py` + `reward_service.py`）：
- **格式分 (format_reward)**：包含 `<think>...</think>` 和 ` ```python ` 代码块 → 0.1 分
- **执行分 (exec_score)**：提取代码 → 执行 → 与测试用例比对 → 0.0~1.0 分
- **总分 = 执行分 + 格式分**（满分 1.1）

---

## 二、核心结果

| 指标 | baseline | v2 | v3 |
|---|---|---|---|
| **总分 (avg_score)** | 0.0680 | 0.0920 | **0.0970** |
| **格式分 (format)** | 0.0680 | 0.0920 | **0.0970** |
| **执行分 (exec)** | 0.0000 | 0.0000 | 0.0000 |
| 格式合规率 | 68% | 92% | **97%** |
| 平均响应长度 | 1497 chars | 991 chars | **788 chars** |

---

## 三、关键发现

### 3.1 训练只学到了格式，没学到代码能力

三个模型的执行分全部为 0。总分的提升完全来自格式合规率的提高：
- baseline 68% → v2 92% → v3 97%

**这意味着 Stage1 的 GRPO 训练本质上只教会了模型"按格式输出"，没有提升代码正确性。**

### 3.2 执行分为 0 的根本原因：reward 评分逻辑存在缺陷

经过排查，`reward_service.py` 的评分逻辑对大部分题目无法正确评分：

**问题 1：函数调用型（770/1000 题）— class 方法无法识别**

`_execute_func` 用 `re.search(r"^def\s+(\w+)\s*\(", code, re.MULTILINE)` 提取函数名。但模型生成的代码大量是 `class Solution` 内的方法（缩进的 `def`），正则匹配不到。

```python
# 模型生成的典型代码：
class Solution:
    def validMountainArray(self, A):  # ← ^def 匹配不到，因为前面有缩进
        ...
```

**问题 2：stdin 型（230/1000 题）— 模型不生成 I/O 代码**

`execute_code` 通过 stdin/stdout 比对，要求代码自己读 `input()` 并 `print()` 结果。但模型生成的是纯函数定义，没有 I/O 调用。

```python
# 模型生成的：
def DNA_strand(dna):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[c] for c in dna)
# ← 没有 print(DNA_strand(input()))，所以 stdout 为空
```

**问题 3：expected_output 格式不匹配**

stdin 型的 `tc["output"]` 是列表（如 `['TTTT']`），但代码中用 `str(tc["output"])` 转换后变成 `"['TTTT']"`，与实际输出 `"TTTT"` 永远不匹配。

### 3.3 训练时也存在同样的问题

从 v2 的 TensorBoard 日志验证：
- score 范围：0.046 ~ 0.106
- 62 步中只有 1 步 score > 0.1（即只有 1 步有非零 exec_score）
- **训练过程中模型几乎从未获得过代码执行的正向奖励**

这意味着 GRPO 训练的梯度信号几乎全部来自 format_reward，模型只学到了格式规范。

### 3.4 格式顺序的变化

Prompt 要求的格式是：先 `<think>` 再 ` ```python `。

| 模型 | think 在前 | code 在前 |
|---|---|---|
| baseline | 26 | 50 |
| v2 | 29 | 67 |
| v3 | **71** | 27 |

- baseline 和 v2 大量出现 code block 在 think 之前（甚至 think 嵌在 code block 内部）
- v3 训练了 3 个 epoch 后，格式顺序明显改善（71% 正确顺序）
- 但 v2 的格式顺序反而比 baseline 更差，说明 v2 只学到了"包含这两个标记"，没学到正确顺序

### 3.5 响应长度持续缩短

baseline 1497 → v2 991 → v3 788 chars

训练确实让模型学会了更简洁的输出（overlong_penalty 起了作用），但代码质量没有提升。

---

## 四、结论

### 4.1 Stage1 训练的实际效果

| 维度 | 效果 |
|---|---|
| 格式合规 | ✅ 有效（68% → 97%） |
| 响应长度控制 | ✅ 有效（1497 → 788 chars） |
| 代码正确性 | ❌ 无效（exec_score 始终为 0） |

**Stage1 训练只完成了"格式对齐"，没有完成"能力提升"。**

### 4.2 根因

**reward 评分函数存在严重 bug**，导致训练过程中模型几乎无法获得代码执行的正向奖励。GRPO 没有有效的梯度信号来优化代码质量。

---

## 五、修复建议（Stage2 前必须完成）

### 5.1 修复 `_execute_func`：支持 class 方法

```python
# 当前（只匹配顶层 def）：
m = re.search(r"^def\s+(\w+)\s*\(", code_str, re.MULTILINE)

# 修复后（匹配任意 def，包括 class 内的方法）：
m = re.search(r"class\s+(\w+)", code_str)
if m:
    class_name = m.group(1)
    # 找 class 内的第一个方法
    m2 = re.search(r"def\s+(\w+)\s*\(self", code_str)
    if m2:
        func_name = m2.group(1)
        # 调用方式：ClassName().method(*args)
```

### 5.2 修复 stdin 型评分：自动包装函数调用

对于只定义了函数但没有 I/O 的代码，自动添加调用逻辑：

```python
# 检测到代码只有函数定义时，自动包装：
code_str += f"\nimport sys\nprint({func_name}(sys.stdin.read().strip()))"
```

### 5.3 修复 expected_output 格式

```python
# 当前：
execute_code(code, tc["input"][0], str(tc["output"]))
# str(['TTTT']) → "['TTTT']"  ← 错误

# 修复：
expected = tc["output"][0] if isinstance(tc["output"], list) else tc["output"]
execute_code(code, tc["input"][0], str(expected))
```

### 5.4 添加评分验证

在训练前，先用评分函数对一批样本进行测试，确认能产生非零的 exec_score。

---

## 六、下一步计划

1. **修复 reward 评分逻辑**（以上 5.1-5.4）
2. **验证修复后的评分**：对 baseline 模型重新评测，确认 exec_score > 0
3. **准备独立测试集**：从训练集外获取新题目
4. **重新训练 Stage1**：使用修复后的 reward，从原始模型开始
5. **评测确认**：确认代码正确性有实质提升后，再进入 Stage2
