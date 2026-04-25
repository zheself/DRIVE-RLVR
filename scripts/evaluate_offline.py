#!/usr/bin/env python3
"""
模型评估脚本（离线版）

直接使用 reward_verifier.py 中的评分逻辑，不依赖 reward service。
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# 添加项目根目录到 path，以便导入 reward_verifier
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reward_verifier import format_reward, execute_code
from reward_service import _compute, _is_stdin_type, _execute_func


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


def evaluate_sample(model, tokenizer, sample: Dict[str, Any], max_new_tokens: int = 768) -> Dict[str, Any]:
    prompt = sample["prompt"]
    test_cases_raw = sample["test_cases"]
    test_cases = json.loads(test_cases_raw) if isinstance(test_cases_raw, str) else test_cases_raw

    response = generate_response(model, tokenizer, prompt, max_new_tokens)

    # 直接调用 _compute 评分
    score = _compute(response, test_cases)
    fmt_score = format_reward(response)

    return {
        "prompt": prompt[:100] + "...",
        "response": response,
        "response_length": len(response),
        "score": score,
        "format_score": fmt_score,
        "exec_score": score - fmt_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model (offline, no reward service needed)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.device)

    print(f"Loading test data from {args.test_data}...")
    test_samples = []
    with open(args.test_data, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.max_samples:
                break
            test_samples.append(json.loads(line))
    print(f"Loaded {len(test_samples)} test samples")

    results = []
    total_score = 0.0
    total_fmt = 0.0
    total_exec = 0.0

    for sample in tqdm(test_samples, desc="Evaluating"):
        result = evaluate_sample(model, tokenizer, sample, args.max_new_tokens)
        results.append(result)
        total_score += result["score"]
        total_fmt += result["format_score"]
        total_exec += result["exec_score"]

    n = len(results)
    avg_score = total_score / n
    avg_fmt = total_fmt / n
    avg_exec = total_exec / n
    avg_len = sum(r["response_length"] for r in results) / n

    # 分数分布
    zero_exec = sum(1 for r in results if r["exec_score"] == 0)
    partial_exec = sum(1 for r in results if 0 < r["exec_score"] < 1)
    full_exec = sum(1 for r in results if r["exec_score"] >= 1.0)
    has_format = sum(1 for r in results if r["format_score"] > 0)

    summary = {
        "model_path": args.model_path,
        "num_samples": n,
        "avg_score": avg_score,
        "avg_format_score": avg_fmt,
        "avg_exec_score": avg_exec,
        "avg_response_length": avg_len,
        "format_compliance_rate": has_format / n,
        "zero_exec_count": zero_exec,
        "partial_exec_count": partial_exec,
        "full_exec_count": full_exec,
    }

    output_data = {"summary": summary, "results": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Model: {args.model_path}")
    print(f"Samples: {n}")
    print(f"{'='*70}")
    print(f"  Avg total score:  {avg_score:.4f}")
    print(f"  Avg format score: {avg_fmt:.4f}  (compliance: {has_format}/{n} = {has_format/n:.0%})")
    print(f"  Avg exec score:   {avg_exec:.4f}  (full: {full_exec}, partial: {partial_exec}, zero: {zero_exec})")
    print(f"  Avg response len: {avg_len:.0f} chars")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
