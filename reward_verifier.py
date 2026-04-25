"""
reward_verifier.py
RLVR 训练核心组件：奖励验证器
用于评估模型生成的 Python 代码的正确性与格式规范性
"""

import re
import subprocess
import tempfile
import os
import resource


def _float_tolerant_match(actual: str, expected: str, tol: float = 1e-4) -> bool:
    """逐行浮点容差比较，用于 competition 题目中浮点输出的匹配"""
    actual_lines = actual.split('\n')
    expected_lines = expected.split('\n')
    if len(actual_lines) != len(expected_lines):
        return False
    for a, e in zip(actual_lines, expected_lines):
        a, e = a.strip(), e.strip()
        if a == e:
            continue
        try:
            if abs(float(a) - float(e)) > tol:
                return False
        except ValueError:
            return False
    return True


def execute_code(code_str: str, input_data: str, expected_output: str) -> float:
    """
    在沙箱中执行模型生成的代码，并与期望输出比较。

    参数:
        code_str: 模型生成的 Python 代码字符串
        input_data: 测试用例的标准输入
        expected_output: 期望的标准输出

    返回:
        1.0 — 输出匹配；0.0 — 超时、报错或不匹配
    """
    # 将代码写入临时文件
    tmp = tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False)
    try:
        tmp.write(code_str)
        tmp.close()

        def _set_limits():
            mem = 512 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

        result = subprocess.run(
            ["python3", tmp.name],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=5.0,
            preexec_fn=_set_limits,
        )

        if result.returncode != 0:
            return 0.0
        actual = result.stdout.strip()
        expected = expected_output.strip()
        if actual == expected:
            return 1.0
        if _float_tolerant_match(actual, expected):
            return 1.0
        return 0.0

    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0
    finally:
        os.unlink(tmp.name)


def format_reward(response_text: str) -> float:
    """
    检查回复是否符合格式要求：包含 <think>...</think> 标签和 ```python 代码块。

    参数:
        response_text: 模型生成的完整回复字符串

    返回:
        0.1 — 格式合规；0.0 — 格式不符
    """
    has_think = "<think>" in response_text and "</think>" in response_text
    has_code_block = "```python" in response_text
    return 0.1 if (has_think and has_code_block) else 0.0


def compute_reward(response_text: str, test_cases: list) -> float:
    """
    综合打分接口，作为 OpenRLHF 自定义奖励函数的主入口。

    评分规则:
        - 执行分 = 通过用例数 / 总用例数 (0.0~1.0)
        - 格式分 = 0.1（含 <think> 和 ```python 块）
        - Total = 执行分 + 格式分

    参数:
        response_text: 模型生成的完整回复字符串
        test_cases: [{"input": ..., "output": ...}, ...] 列表

    返回:
        最终奖励分数 (0.0 ~ 1.1)
    """
    fmt_score = format_reward(response_text)

    match = re.search(r"```python\s*(.*?)```", response_text, re.DOTALL)
    if not match:
        return fmt_score

    if not test_cases:
        return fmt_score

    code_str = match.group(1).strip()
    try:
        passed = sum(
            execute_code(code_str, tc["input"], tc["output"])
            for tc in test_cases
        )
        exec_score = passed / len(test_cases)
    except Exception:
        exec_score = 0.0

    return exec_score + fmt_score
