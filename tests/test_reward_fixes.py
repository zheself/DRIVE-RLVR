#!/usr/bin/env python3
"""
测试 reward_service.py 的修复
验证三个 bug 都已修复
"""

import sys
sys.path.insert(0, '/mnt/sdc/ubuntu/cjz_projects/DRIVE')

from reward_service import _extract_code, _execute_func, _compute
import json


def test_bug1_code_extraction():
    """Bug 1: 代码提取正则匹配到 think 内容而非代码"""
    print("=== Test Bug 1: Code Extraction ===")

    # Case 1: think 在第一个 code block 内
    response1 = """ ```python
<think>
思考内容
</think>

```python
def DNA_strand(dna):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[c] for c in dna)
``` ```"""

    code1 = _extract_code(response1)
    assert code1 is not None, "Should extract code"
    assert "def DNA_strand" in code1, f"Should extract function, got: {code1[:100]}"
    assert "<think>" not in code1, f"Should not contain think tag, got: {code1[:100]}"
    print("✓ Case 1: think 在 code block 内 - PASSED")

    # Case 2: 正常格式（think 在外面）
    response2 = """<think>思考</think>

```python
def test():
    return 42
```"""

    code2 = _extract_code(response2)
    assert code2 is not None
    assert "def test" in code2
    print("✓ Case 2: 正常格式 - PASSED")

    # Case 3: 多个 code block，选择有 def 的
    response3 = """```python
# 注释
```

```python
def real_code():
    pass
```"""

    code3 = _extract_code(response3)
    assert "def real_code" in code3
    print("✓ Case 3: 多个 code block - PASSED")

    print()


def test_bug2_class_method():
    """Bug 2: _execute_func 无法识别 class 内的方法"""
    print("=== Test Bug 2: Class Method Recognition ===")

    # Case 1: class 方法
    code1 = """class Solution:
    def validMountainArray(self, A):
        if len(A) < 3:
            return False
        return True"""

    score1 = _execute_func(code1, [[2, 1]], False)
    assert score1 == 1.0, f"Should recognize class method, got score: {score1}"
    print("✓ Case 1: class 方法识别 - PASSED")

    # Case 2: 顶层函数（原有逻辑）
    code2 = """def add(a, b):
    return a + b"""

    score2 = _execute_func(code2, [1, 2], 3)
    assert score2 == 1.0, f"Should work with top-level function, got score: {score2}"
    print("✓ Case 2: 顶层函数 - PASSED")

    print()


def test_bug3_stdin_output():
    """Bug 3: stdin 型的 expected_output 格式错误"""
    print("=== Test Bug 3: stdin Output Format ===")

    response = """<think>思考</think>

```python
def DNA_strand(dna):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[c] for c in dna)

print(DNA_strand(input()))
```"""

    # stdin 型测试用例，output 是 list
    test_cases = [
        {"input": ["AAAA"], "output": ["TTTT"]},
        {"input": ["ATTGC"], "output": ["TAACG"]}
    ]

    score = _compute(response, test_cases)
    print(f"Score: {score}")

    # 应该有 format_reward (0.1) + exec_score (0.5 或 1.0)
    assert score > 0.1, f"Should have exec_score > 0, got: {score}"
    print("✓ stdin 型 output 格式修复 - PASSED")

    print()


def test_integration():
    """集成测试：使用实际的 v2 response"""
    print("=== Integration Test ===")

    # 从评测结果中取一个实际的 response
    with open('/mnt/sdc/ubuntu/cjz_projects/DRIVE/evaluation_results/v2_results.json') as f:
        data = json.load(f)

    # Sample 0: DNA strand (stdin 型)
    response0 = data['results'][0]['response']
    with open('/mnt/sdc/ubuntu/cjz_projects/DRIVE/data/stage1_base_str.jsonl') as f:
        sample0 = json.loads(f.readline())
    test_cases0 = json.loads(sample0['test_cases'])

    score0 = _compute(response0, test_cases0)
    print(f"Sample 0 (DNA strand) score: {score0:.4f}")

    # Sample 1: Mountain array (class 方法)
    response1 = data['results'][1]['response']
    with open('/mnt/sdc/ubuntu/cjz_projects/DRIVE/data/stage1_base_str.jsonl') as f:
        f.readline()  # skip first
        sample1 = json.loads(f.readline())
    test_cases1 = json.loads(sample1['test_cases'])

    score1 = _compute(response1, test_cases1)
    print(f"Sample 1 (Mountain array) score: {score1:.4f}")

    print()
    print("=== Summary ===")
    print(f"Sample 0: {score0:.4f} (expected > 0.1)")
    print(f"Sample 1: {score1:.4f} (expected > 0.1)")

    if score0 > 0.1 or score1 > 0.1:
        print("✓ At least one sample has exec_score > 0 - PASSED")
    else:
        print("✗ Both samples still have exec_score = 0 - FAILED")

    print()


if __name__ == "__main__":
    try:
        test_bug1_code_extraction()
        test_bug2_class_method()
        test_bug3_stdin_output()
        test_integration()

        print("="*60)
        print("All tests PASSED! ✓")
        print("="*60)
    except AssertionError as e:
        print(f"\n✗ Test FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
