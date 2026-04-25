"""
reward_service.py
Flask 服务，将奖励计算暴露为 OpenRLHF remote_rm_url 端点。
支持两种测试用例格式：
  - stdin型: input=[str], output=str  → subprocess stdin/stdout 比较
  - 函数调用型: input=[*args], output=any → 提取函数并调用比较
"""

import json
import re
import subprocess
import tempfile
import os
import resource
from flask import Flask, request, jsonify
from reward_verifier import format_reward, execute_code

app = Flask(__name__)


def _extract_code(response_text: str) -> str | None:
    """从 response 中提取代码，处理 think 嵌入 code block 的情况"""
    # 策略：找所有 ```python 开始位置和所有 ``` 结束位置，配对提取
    python_starts = [(m.start(), m.end()) for m in re.finditer(r"```python", response_text)]
    code_ends = [m.start() for m in re.finditer(r"```(?!\w)", response_text)]  # ``` 后面不跟字母

    if not python_starts or not code_ends:
        return None

    # 为每个 ```python 找最近的后续 ```，且不能被之前的 ```python 使用过
    blocks = []
    used_ends = set()

    for start_pos, start_end in python_starts:
        # 找第一个在 start_end 之后且未被使用的 ```
        for end_pos in code_ends:
            if end_pos > start_end and end_pos not in used_ends:
                code = response_text[start_end:end_pos].strip()
                # 清理末尾可能残留的 ```
                code = re.sub(r'```\s*$', '', code).strip()
                blocks.append(code)
                used_ends.add(end_pos)
                break

    if not blocks:
        return None

    # 优先选包含 def/class 且不包含 think 的块
    for block in blocks:
        if re.search(r"(?:^def\s|class\s)", block, re.MULTILINE) and "<think>" not in block:
            return block

    # 其次选包含 def/class 的块（即使有 think）
    for block in blocks:
        if re.search(r"(?:^def\s|class\s)", block, re.MULTILINE):
            return block

    # 都不包含时取最后一个（跳过 think 块）
    for block in reversed(blocks):
        if "<think>" not in block:
            return block

    return blocks[-1] if blocks else None


def _is_stdin_type(tc: dict) -> bool:
    # 如果有 fn_name 字段，明确是函数调用型
    if "fn_name" in tc:
        return False
    inp = tc["input"]
    # 纯字符串输入才是 stdin 型（APPS stdin 格式）
    if isinstance(inp, str):
        return True
    # 没有 fn_name 且 input 是 list → 函数调用型
    return False


def _execute_func(code_str: str, args: list, expected) -> float:
    """提取代码中的函数，用 args 调用，与 expected 比较。"""
    # APPS fn_name 型的 output 经常包在 list 里，如 ['TTTT']，需要解包
    if isinstance(expected, list) and len(expected) == 1:
        expected = expected[0]

    # 1. 尝试匹配顶层 def
    m = re.search(r"^def\s+(\w+)\s*\(", code_str, re.MULTILINE)
    if m:
        func_name = m.group(1)
        call_expr = f"{func_name}(*_args)"
    else:
        # 2. 尝试匹配 class 内的方法
        cm = re.search(r"^class\s+(\w+)", code_str, re.MULTILINE)
        mm = re.search(r"^\s+def\s+(\w+)\s*\(self", code_str, re.MULTILINE)
        if cm and mm:
            class_name = cm.group(1)
            method_name = mm.group(1)
            call_expr = f"{class_name}().{method_name}(*_args)"
        else:
            return 0.0

    test_snippet = f"""
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict, Counter, deque
from itertools import permutations, combinations
from functools import lru_cache
import math, re, heapq, bisect

{code_str}

import json, sys
_args = json.loads(sys.stdin.read())
_result = {call_expr}
print(json.dumps(_result))
"""
    tmp = tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False)
    try:
        tmp.write(test_snippet)
        tmp.close()

        def _set_limits():
            mem = 512 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

        result = subprocess.run(
            ["python3", tmp.name],
            input=json.dumps(args),
            capture_output=True,
            text=True,
            timeout=5.0,
            preexec_fn=_set_limits,
        )
        if result.returncode != 0:
            return 0.0
        actual = json.loads(result.stdout.strip())
        return 1.0 if actual == expected else 0.0
    except Exception:
        return 0.0
    finally:
        os.unlink(tmp.name)


def _compute(response_text: str, test_cases: list) -> float:
    fmt = format_reward(response_text)
    code = _extract_code(response_text)
    if not code or not test_cases:
        return fmt

    scores = []
    for tc in test_cases:
        if _is_stdin_type(tc):
            # stdin 型：input 是纯字符串，output 也是纯字符串
            code_to_run = code
            if 'input()' not in code and 'sys.stdin' not in code:
                # 尝试找函数名并自动包装
                m = re.search(r"^def\s+(\w+)\s*\(", code, re.MULTILINE)
                if m:
                    func_name = m.group(1)
                    code_to_run = f"{code}\nprint({func_name}(input()))"

            input_str = tc["input"]
            expected = tc["output"]
            if isinstance(expected, list) and len(expected) == 1:
                expected = expected[0]
            expected_str = str(expected).strip() if expected is not None else "None"
            scores.append(execute_code(code_to_run, input_str, expected_str))
        else:
            scores.append(_execute_func(code, tc["input"], tc["output"]))

    return sum(scores) / len(scores) + fmt


@app.route("/scores", methods=["POST"])
def scores():
    data = request.get_json()
    queries = data["query"]
    prompts = data["prompts"]
    labels = data["labels"]

    result = []
    for query, prompt, label in zip(queries, prompts, labels):
        response_text = query[len(prompt):] if query.startswith(prompt) else query
        test_cases = json.loads(label) if isinstance(label, str) else label
        result.append(_compute(response_text, test_cases))

    # OpenRLHF 每次发送单条请求，响应体直接作为 rewards_info 使用
    # 需要返回标量 rewards 和 scores，不能是列表
    score = result[0]
    return jsonify({"rewards": score, "scores": score})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
