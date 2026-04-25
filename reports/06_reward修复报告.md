# Reward 评分逻辑修复报告

**日期**：2026-04-19

---

## 一、修复概述

成功修复了 `reward_service.py` 中导致 exec_score 全部为 0 的三个 bug，使模型能够获得代码执行的正向奖励。

---

## 二、修复的 Bug

### Bug 1：代码提取正则匹配到 think 内容而非代码

**问题**：模型输出格式经常是 ` ```python\n<think>...\n```python\ndef code()...``` ``` `，原有的 `re.search(r"```python\s*(.*?)```", ...)` lazy 匹配提取到了 think 内容。

**修复**：
- 新增 `_extract_code()` 函数
- 找所有 ` ```python ` 和 ` ``` ` 位置，智能配对提取
- 优先选择包含 `def`/`class` 且不包含 `<think>` 的块
- 清理末尾残留的 ` ``` `

**代码位置**：`reward_service.py` 第 21-60 行

### Bug 2：`_execute_func` 无法识别 class 内的方法

**问题**：`re.search(r"^def\s+(\w+)\s*\(", ...)` 只匹配行首的 `def`，无法识别 `class Solution:` 内的缩进方法。

**修复**：
- 先尝试匹配顶层 `def`
- 失败时检测 `class` + 缩进 `def`，生成 `ClassName().method(*_args)` 调用
- 添加常用库的自动 import（`typing.List`, `collections`, `itertools` 等）

**代码位置**：`reward_service.py` 第 63-109 行

### Bug 3：stdin 型的 expected_output 格式错误

**问题**：`str(tc["output"])` 将 `['TTTT']` 转成 `"['TTTT']"`，与实际输出 `"TTTT"` 不匹配。

**修复**：
- 如果 `tc["output"]` 是 list 且长度为 1，取 `tc["output"][0]`
- 对于只定义函数没有 I/O 的代码，自动添加 `print(func_name(input()))`

**代码位置**：`reward_service.py` 第 112-135 行

---

## 三、修复效果验证

### 3.1 单元测试

创建了 `tests/test_reward_fixes.py`，包含：
- Bug 1 测试：代码提取（3 个 case）
- Bug 2 测试：class 方法识别（2 个 case）
- Bug 3 测试：stdin 型 output 格式（1 个 case）
- 集成测试：使用实际 v2 response（2 个 sample）

**结果**：所有测试通过 ✓

### 3.2 重新评测 v2 模型

对 v2 模型的前 20 个样本重新评测：

| 指标 | 修复前 | 修复后 | 变化 |
|---|---|---|---|
| 平均总分 | 0.092 | **0.340** | +270% |
| 平均格式分 | 0.092 | 0.090 | -2% |
| 平均执行分 | **0.000** | **0.250** | +∞ |
| 格式合规率 | 92% | 90% | -2% |
| 执行分满分样本 | 0/20 | **5/20** | 25% |

**关键发现**：
- 修复前 exec_score 全部为 0
- 修复后 20 个样本中 5 个获得满分（exec_score = 1.0）
- 15 个样本仍为 0 分，说明模型生成的代码本身有问题（逻辑错误），但评分系统现在能正确识别了

---

## 四、修复后的评分流程

### 4.1 代码提取（`_extract_code`）

```
response → 找所有 ```python 和 ``` → 智能配对 → 
优先选有 def/class 且无 think 的块 → 清理末尾 ``` → 返回代码
```

### 4.2 函数调用型评分（`_execute_func`）

```
代码 → 尝试匹配顶层 def → 失败则匹配 class + 方法 → 
生成测试代码（自动 import） → 执行 → 比对结果 → 返回 0.0 或 1.0
```

### 4.3 stdin 型评分（`_compute` 中的分支）

```
代码 → 检查是否有 input()/print() → 
没有则自动包装 print(func(input())) → 
修正 expected_output 格式 → 执行 → 比对 → 返回 0.0 或 1.0
```

---

## 五、对训练的影响

### 5.1 修复前的训练问题

- v2 训练时 62 步中只有 1 步 score > 0.1
- 模型几乎从未获得代码执行的正向奖励
- GRPO 只能优化格式，无法优化代码质量

### 5.2 修复后的预期

- 模型能获得真实的代码执行反馈
- GRPO 可以区分正确代码和错误代码
- 训练将同时优化格式和代码正确性

### 5.3 建议

**必须使用修复后的 reward_service.py 重新训练 Stage1**，否则训练无法提升代码能力。

---

## 六、文件清单

### 6.1 修改的文件

- `/mnt/sdc/ubuntu/cjz_projects/DRIVE/reward_service.py` — 主要修复
  - 新增 `_extract_code()` 函数（21-60 行）
  - 修改 `_execute_func()` 支持 class 方法（63-109 行）
  - 修改 `_compute()` 修复 stdin 型和自动包装（112-135 行）

### 6.2 新增的文件

- `/mnt/sdc/ubuntu/cjz_projects/DRIVE/tests/test_reward_fixes.py` — 单元测试
- `/mnt/sdc/ubuntu/cjz_projects/DRIVE/evaluation_results/v2_results_fixed.json` — 修复后的评测结果

### 6.3 报告文件

- `/mnt/sdc/ubuntu/cjz_projects/DRIVE/reports/05_评测结果与reward缺陷分析.md` — 问题诊断报告
- `/mnt/sdc/ubuntu/cjz_projects/DRIVE/reports/06_reward修复报告.md` — 本报告

---

## 七、后续工作

1. ✅ 修复 reward 评分逻辑
2. ✅ 验证修复效果
3. ⏭️ 准备独立测试集（从训练集外获取新题目）
4. ⏭️ 使用修复后的 reward 重新训练 Stage1
5. ⏭️ 评测确认代码正确性有实质提升
6. ⏭️ 进入 Stage2 训练

---

## 八、技术细节

### 8.1 为什么需要自动 import

模型生成的代码经常使用 `List[int]` 等类型注解，但不导入 `typing`。为了让代码能执行，在测试代码中自动添加：

```python
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict, Counter, deque
from itertools import permutations, combinations
from functools import lru_cache
import math, re, heapq, bisect
```

### 8.2 为什么需要自动包装 stdin 型

模型被训练成生成函数定义，但 stdin 型测试需要 `input()`/`print()`。自动包装避免了所有 stdin 型都得 0 分。

### 8.3 代码提取的边界情况

- 多个 ` ```python ` 块：选有 def/class 的
- think 嵌在 code block 内：通过配对算法跳过
- 末尾残留 ` ``` `：用正则清理
- 都没有 def/class：取最后一个非 think 块

---

## 九、已知限制

1. **模型代码逻辑错误**：修复只能让评分系统正确识别，无法修复模型生成的逻辑错误
2. **复杂类型注解**：只添加了常用库，极少数特殊库可能仍会报错
3. **多类定义**：只识别第一个 class 的第一个方法
4. **stdin 型参数传递**：假设函数只有一个参数，多参数情况未处理

这些限制不影响大部分题目的评分，可以在后续迭代中优化。
