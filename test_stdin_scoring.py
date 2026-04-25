#!/usr/bin/env python3
"""
测试 stdin 型样本的评分逻辑是否正确
从 test.jsonl 中抽取几个 introductory stdin 型样本进行验证
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward_service import _compute, _is_stdin_type

TEST_FILE = "/mnt/sdc/ubuntu/cjz_projects/datasets/codeparrot/apps/test.jsonl"

def get_test_cases(r):
    """从 APPS 记录中提取测试用例（与 prepare_data.py 逻辑一致）"""
    io = json.loads(r["input_output"])
    ins, outs = io["inputs"], io["outputs"]
    fn_name = io.get("fn_name", None)
    cases = []
    for i in range(min(5, len(ins))):
        inp = ins[i]
        out = outs[i]
        # stdin 型中部分 input/output 是 list，归一化为字符串
        if not fn_name:
            if isinstance(inp, list):
                inp = "\n".join(str(x) for x in inp)
            if isinstance(out, list):
                out = "\n".join(str(x) for x in out)
        tc = {"input": inp, "output": out}
        if fn_name:
            tc["fn_name"] = fn_name
        cases.append(tc)
    return cases

def make_prompt(question):
    return (
        "请用 Python 解决以下问题，并务必在生成代码前使用 <think> 标签进行思考，"
        "最终代码必须包裹在 ```python 和 ``` 之间。\n\n问题描述：\n" + question
    )

# 收集 introductory stdin 型样本
stdin_samples = []
with open(TEST_FILE) as f:
    for line in f:
        r = json.loads(line)
        if r.get("difficulty") != "introductory":
            continue

        io = json.loads(r.get("input_output", "{}"))
        ins = io.get("inputs", [])
        outs = io.get("outputs", [])

        # 验证数据有效性
        if len(ins) != len(outs) or len(ins) == 0:
            continue

        # 只要 stdin 型（没有 fn_name）
        if io.get("fn_name"):
            continue

        stdin_samples.append(r)

        if len(stdin_samples) >= 5:
            break

print(f"找到 {len(stdin_samples)} 个 introductory stdin 型样本\n")

# 测试每个样本的评分逻辑
for i, sample in enumerate(stdin_samples, 1):
    question = sample.get("question", "")[:100]
    test_cases = get_test_cases(sample)

    print(f"{'='*70}")
    print(f"样本 {i}: {question}...")
    print(f"测试用例数: {len(test_cases)}")

    # 验证分类逻辑
    is_stdin = _is_stdin_type(test_cases[0])
    print(f"分类结果: {'stdin 型' if is_stdin else 'fn_name 型'} ✓" if is_stdin else "分类结果: fn_name 型 ✗ (应该是 stdin 型)")

    # 显示第一个测试用例
    tc = test_cases[0]
    print(f"\n第一个测试用例:")
    print(f"  input type: {type(tc['input']).__name__}")
    print(f"  input preview: {repr(tc['input'][:50]) if len(tc['input']) > 50 else repr(tc['input'])}")
    print(f"  output type: {type(tc['output']).__name__}")
    print(f"  output: {repr(tc['output'])}")
    print(f"  has fn_name: {'fn_name' in tc}")

    # 构造一个简单的正确响应来测试评分
    # 注意：这里我们不实际运行模型，只是测试评分函数能否正常工作
    mock_response = """<think>
这是一个测试
</think>

```python
def solution():
    return "test"
```"""

    try:
        score = _compute(mock_response, test_cases)
        print(f"\n评分测试: score={score:.2f} (格式分 0.1 + 执行分 {score-0.1:.2f})")
        print("✓ 评分函数运行正常（未报错）")
    except Exception as e:
        print(f"\n✗ 评分函数报错: {e}")

    print()

print(f"{'='*70}")
print("\n结论:")
print("1. 如果所有样本都正确分类为 stdin 型 → 分类逻辑正确")
print("2. 如果评分函数都能正常运行 → 基础逻辑没问题")
print("3. 执行分为 0 是正常的（mock_response 是随机代码，不会通过测试）")
print("\n下一步: 用真实模型生成响应，验证能否获得非零 exec_score")
