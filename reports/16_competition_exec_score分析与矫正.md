# Competition Exec Score 接近 0 的原因分析与矫正方案

## 一、问题描述

Stage2 训练过程中，score 始终在 0.07-0.09 范围内，几乎全由格式分（0.1）贡献，exec_score 接近 0。这意味着模型在 competition 题目上几乎无法生成正确代码。

## 二、根因分析

### 2.1 评分系统对参考解法的测试

为了区分"模型能力不足"和"评分系统有 bug"，我们用 APPS 原始数据中的**人类参考解法**测试了评分系统：

| 结果 | 数量 | 占比 |
|------|------|------|
| Full pass (exec=1.0) | 34/50 | 68% |
| Partial pass (0<exec<1) | 9/50 | 18% |
| Zero pass (exec=0) | 7/50 | 14% |

**参考解法（已知正确的代码）在评分系统中只有 68% 的 full pass 率。** 这意味着评分系统本身就有 32% 的误判率。

### 2.2 参考解法失败原因分类

| 失败原因 | 数量 | 占比 | 说明 |
|----------|------|------|------|
| **Wrong Answer** | 9 | 56% | 参考解法本身有 bug，或 APPS 数据集标注错误 |
| **Runtime Error** | 3 | 19% | 内存限制 256MB 不够，或 Python 版本差异 |
| **Precision Mismatch** | 3 | 19% | 输出 `1.0` vs 期望 `1.000000000000000`，精确字符串匹配失败 |
| **Timeout** | 1 | 6% | 2 秒超时不够 |

### 2.3 三层问题叠加

Competition 上 exec_score ≈ 0 是三个问题叠加的结果：

**第一层：评分系统缺陷（可修复，预计提升 10-15%）**

1. **浮点精度匹配过严**：当前使用 `stdout.strip() == expected.strip()` 精确匹配。`1.0` ≠ `1.000000000000000` 会被判错。Competition 中 6% 的测试用例涉及浮点输出。
2. **内存限制过低**：256MB 对部分 competition 题目不够（图论、大数组）。
3. **超时过短**：2 秒对 Python 执行 competition 级别算法偏紧，尤其是涉及大输入的题目。

**第二层：数据集噪声（难以修复）**

- APPS 数据集中约 18% 的 competition 参考解法本身就是错的（wrong answer）。这些错误的标注会产生错误的训练信号——模型即使生成了正确代码，也可能因为标注错误而得到 0 分。

**第三层：模型能力天花板（根本限制）**

- Competition 题目需要高级算法（图论、DP、线段树、数论等），3B 模型在这些方面的能力远不如 7B/70B。
- 即使评分系统完美，3B 模型在 competition 上的 exec_score 也不会很高。
- 对比：introductory 上 baseline exec=0.465，competition 上即使修复评分也预计 < 0.1。

## 三、矫正方案

### 方案 1：修复评分系统（推荐，立即可做）

修改 `reward_service.py` 中的 `execute_code` 函数：

```python
def execute_code(code_str, input_data, expected_output):
    # ... 执行代码 ...
    
    actual = result.stdout.strip()
    expected = expected_output.strip()
    
    # 1. 精确匹配
    if actual == expected:
        return 1.0
    
    # 2. 浮点容差匹配（新增）
    try:
        actual_lines = actual.split('\n')
        expected_lines = expected.split('\n')
        if len(actual_lines) == len(expected_lines):
            all_match = True
            for a, e in zip(actual_lines, expected_lines):
                if a.strip() != e.strip():
                    try:
                        if abs(float(a) - float(e)) > 1e-6:
                            all_match = False
                            break
                    except ValueError:
                        all_match = False
                        break
            if all_match:
                return 1.0
    except:
        pass
    
    return 0.0
```

同时调整资源限制：
- 内存：256MB → 512MB
- 超时：2s → 5s（competition 专用，introductory 保持 2s）

**预期效果**：参考解法的 full pass 率从 68% 提升到 ~80-85%。

### 方案 2：清洗 Competition 数据（推荐，需要一次性工作）

用参考解法预跑所有 competition 样本，过滤掉参考解法都通不过的题目：

```python
# 过滤逻辑
valid_samples = []
for sample in competition_samples:
    ref_solution = sample['solutions'][0]
    score = evaluate(ref_solution, sample['test_cases'])
    if score >= 1.0:  # 只保留参考解法能全通过的题目
        valid_samples.append(sample)
```

预计从 360 题中保留 ~245 题（68%），但这些题目的评分信号是可靠的。

### 方案 3：降低 Stage2 难度（推荐）

用 **interview** 难度替代 competition：
- interview 题目对 3B 模型更可行（baseline 在 introductory 上 exec=0.465，interview 预计 0.1-0.2）
- 训练信号更丰富（更多非零 exec_score）
- GRPO 需要组内方差才能产生有效梯度，全 0 的 exec_score 没有区分度

### 方案 4：混合难度训练（可选）

Stage2 数据混合 interview + competition：
- 70% interview（提供丰富的训练信号）
- 30% competition（保持难题暴露）
- 这样 GRPO 的 group_reward_std 不会太低

## 四、优先级建议

| 优先级 | 方案 | 工作量 | 预期收益 |
|--------|------|--------|----------|
| P0 | 修复浮点精度匹配 | 30 分钟 | 消除 6% 的误判 |
| P0 | 清洗 competition 数据 | 1 小时 | 消除 18% 的噪声标注 |
| P1 | 调整超时和内存限制 | 10 分钟 | 消除 8% 的误判 |
| P1 | 用 interview 替代 competition | 2 小时 | 根本解决训练信号稀疏问题 |
| P2 | 混合难度训练 | 3 小时 | 平衡难度梯度 |

## 五、已执行的矫正（2026-04-22）

### 5.1 修复浮点精度匹配（方案 1 ✅ 已完成）

修改 `reward_verifier.py`：

- 新增 `_float_tolerant_match(actual, expected, tol=1e-4)` 函数，在精确匹配失败后逐行尝试浮点容差比较
- 容差 1e-4，覆盖 competition 题目常见的精度要求（如 `5.666666667` vs `5.6666666666666`）
- 只在精确匹配失败时触发，不影响整数/字符串类型的输出比较

同步调整资源限制：
- `reward_verifier.py`：内存 256MB → 512MB，超时 2s → 5s
- `reward_service.py`：`_execute_func` 内存 256MB → 512MB，超时 2s → 5s

**对 stage1 的影响**：无。这些改动是单调放宽——只可能让评分变高或不变，不会让原来通过的 case 失败。stage1 的 introductory 题目输出基本是整数或字符串，精确匹配即可通过，不会触发浮点容差分支。

### 5.2 清洗 Competition 数据（方案 2 ✅ 已完成）

新建脚本 `scripts/clean_competition.py`，用 APPS 原始数据中的参考解法预跑每道 competition 题的 test cases：

| 指标 | 数值 |
|------|------|
| 原始 competition 题目数 | 361 |
| 修复后参考解法 full pass | **302 (83.7%)** |
| 过滤题目数 | 59 (16.3%) |

对比修复前后参考解法 full pass 率：

| 阶段 | Full pass 率 | 说明 |
|------|-------------|------|
| 修复前（报告 §2.1 采样测试） | 68% | 50 题采样 |
| 修复后（全量验证） | **83.7%** | 361 题全量 |
| 提升 | **+15.7pp** | 浮点容差 + 资源限制放宽的效果 |

被过滤的 59 题分布：
- exec_score = 0（参考解法完全跑不过）：29 题 — 数据标注错误或参考解法本身有 bug
- 0 < exec_score < 1（部分通过）：21 题 — 部分 test case 标注有误
- test_cases 无效：9 题

清洗后数据输出：
- `data/stage2_v3.jsonl`（302 samples，test_cases 为 list 格式）
- `data/stage2_v3_str.jsonl`（302 samples，test_cases 为 JSON 字符串格式，供 OpenRLHF 使用）
- `data/stage2_v3_clean_log.json`（清洗详情日志）

## 六、结论

Competition exec_score ≈ 0 不是单一原因，而是**评分缺陷 + 数据噪声 + 模型能力**三层问题叠加。其中评分缺陷和数据噪声已完成修复（§5.1、§5.2），参考解法 full pass 率从 68% 提升到 83.7%，清洗后保留 302 道可靠题目。但 3B 模型在 competition 上的能力天花板仍然很低，建议将 stage2 的重心转向 interview 难度。
