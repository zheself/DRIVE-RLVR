#!/usr/bin/env python3
"""
HumanEval 评测脚本

HumanEval 格式：
  - prompt: 函数签名 + docstring（模型需要补全函数体）
  - test: check(candidate) 函数，包含 assert 语句
  - entry_point: 函数名

评测流程：
  1. 用训练时的 prompt 模板包装 HumanEval prompt
  2. 模型生成回复，提取 ```python 代码块
  3. 拼接代码 + test + check(entry_point)，执行判断是否通过
"""

import json
import sys
import argparse
import re
import subprocess
import tempfile
import os
import resource
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reward_verifier import format_reward
from reward_service import _extract_code


PROMPT_TEMPLATE = """请用 Python 解决以下问题，并务必在生成代码前使用 <think> 标签进行思考，最终代码必须包裹在 ```python 和 ``` 之间。

问题描述：
请补全以下 Python 函数的实现：

```python
{prompt}
```

请直接给出完整的函数实现（包含函数签名）。"""


def load_model(model_path: str, device: str = "auto"):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 768) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def execute_humaneval_test(code: str, test_code: str, entry_point: str, timeout: float = 5.0) -> bool:
    """执行 HumanEval 测试：拼接代码 + test + check(entry_point)"""
    full_code = f"""{code}

{test_code}

check({entry_point})
"""
    tmp = tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False)
    try:
        tmp.write(full_code)
        tmp.close()

        def _set_limits():
            mem = 256 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

        result = subprocess.run(
            ["python3", tmp.name],
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_set_limits,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        os.unlink(tmp.name)


def evaluate_sample(model, tokenizer, row, max_new_tokens: int = 768):
    """评测单个 HumanEval 样本"""
    task_id = row["task_id"]
    raw_prompt = row["prompt"]
    test_code = row["test"]
    entry_point = row["entry_point"]

    # 构造模型输入
    model_prompt = PROMPT_TEMPLATE.format(prompt=raw_prompt.strip())
    response = generate_response(model, tokenizer, model_prompt, max_new_tokens)

    # 提取代码
    code = _extract_code(response)
    fmt_score = format_reward(response)

    passed = False
    if code:
        passed = execute_humaneval_test(code, test_code, entry_point)

    return {
        "task_id": task_id,
        "prompt": raw_prompt[:100] + "...",
        "response": response,
        "extracted_code": code or "",
        "response_length": len(response),
        "format_score": fmt_score,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on HumanEval")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to HumanEval parquet file or directory")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=164)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.device)

    # 加载数据
    data_path = args.data_path
    if os.path.isdir(data_path):
        parquet_files = list(Path(data_path).rglob("*.parquet"))
        if not parquet_files:
            print(f"No parquet files found in {data_path}")
            sys.exit(1)
        data_path = str(parquet_files[0])
    print(f"Loading HumanEval data from {data_path}...")
    df = pd.read_parquet(data_path)
    if args.max_samples < len(df):
        df = df.head(args.max_samples)
    print(f"Loaded {len(df)} tasks")

    results = []
    pass_count = 0
    fmt_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        result = evaluate_sample(model, tokenizer, row, args.max_new_tokens)
        results.append(result)
        if result["passed"]:
            pass_count += 1
        if result["format_score"] > 0:
            fmt_count += 1

    n = len(results)
    pass_at_1 = pass_count / n
    fmt_rate = fmt_count / n
    avg_len = sum(r["response_length"] for r in results) / n

    summary = {
        "model_path": args.model_path,
        "dataset": "HumanEval",
        "num_tasks": n,
        "pass@1": pass_at_1,
        "pass_count": pass_count,
        "fail_count": n - pass_count,
        "format_compliance_rate": fmt_rate,
        "avg_response_length": avg_len,
    }

    output_data = {"summary": summary, "results": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: HumanEval ({n} tasks)")
    print(f"{'='*70}")
    print(f"  pass@1:           {pass_at_1:.4f}  ({pass_count}/{n})")
    print(f"  Format compliance: {fmt_rate:.0%}  ({fmt_count}/{n})")
    print(f"  Avg response len:  {avg_len:.0f} chars")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
