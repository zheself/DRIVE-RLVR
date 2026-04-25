#!/usr/bin/env python3
"""
清洗 Competition 数据

用 APPS 原始数据中的参考解法预跑 test cases，
过滤掉参考解法都通不过的题目，确保训练信号可靠。
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reward_verifier import execute_code

APPS_TRAIN = "/mnt/sdc/ubuntu/cjz_projects/datasets/codeparrot/apps/train.jsonl"
OUTPUT_DIR = "/mnt/sdc/ubuntu/cjz_projects/DRIVE/data"


def make_prompt(question):
    return (
        "请用 Python 解决以下问题，并务必在生成代码前使用 <think> 标签进行思考，"
        "最终代码必须包裹在 ```python 和 ``` 之间。\n\n问题描述：\n" + question
    )


def get_test_cases(record):
    io = json.loads(record["input_output"])
    ins, outs = io["inputs"], io["outputs"]
    fn_name = io.get("fn_name", None)
    cases = []
    for i in range(min(5, len(ins))):
        inp, out = ins[i], outs[i]
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


def validate_with_solution(solution: str, test_cases: list) -> dict:
    """用参考解法跑 test cases，返回详细结果"""
    scores = []
    fail_reasons = []

    for tc in test_cases:
        inp = tc["input"]
        expected = tc["output"]
        if isinstance(expected, list) and len(expected) == 1:
            expected = expected[0]
        expected_str = str(expected).strip()

        score = execute_code(solution, inp, expected_str)
        scores.append(score)
        if score == 0.0:
            fail_reasons.append("fail")

    total = sum(scores) / len(scores) if scores else 0.0
    return {
        "exec_score": total,
        "passed": len([s for s in scores if s == 1.0]),
        "total": len(scores),
        "full_pass": total >= 1.0,
    }


def main():
    # 1. 加载原始 APPS competition 数据
    print("Loading APPS train.jsonl...")
    competition_records = []
    with open(APPS_TRAIN) as f:
        for line in f:
            r = json.loads(line)
            if r.get("difficulty") == "competition":
                # 需要有 solutions 和有效的 input_output
                sols = r.get("solutions", "")
                if sols:
                    try:
                        sol_list = json.loads(sols)
                        if sol_list:
                            competition_records.append(r)
                    except:
                        pass

    print(f"Found {len(competition_records)} competition records with solutions")

    # 2. 用参考解法验证每道题
    valid = []
    invalid = []
    stats = defaultdict(int)

    for r in tqdm(competition_records, desc="Validating"):
        try:
            test_cases = get_test_cases(r)
        except Exception:
            stats["invalid_test_cases"] += 1
            invalid.append({"id": r.get("id", "?"), "reason": "invalid_test_cases"})
            continue

        sol_list = json.loads(r["solutions"])
        solution = sol_list[0]

        result = validate_with_solution(solution, test_cases)

        if result["full_pass"]:
            valid.append(r)
            stats["full_pass"] += 1
        else:
            stats["fail"] += 1
            invalid.append({
                "id": r.get("id", "?"),
                "reason": "solution_fail",
                "exec_score": result["exec_score"],
                "passed": result["passed"],
                "total": result["total"],
            })

    # 3. 导出清洗后的数据
    import random
    random.seed(42)
    random.shuffle(valid)

    out_path = os.path.join(OUTPUT_DIR, "stage2_v3.jsonl")
    out_str_path = os.path.join(OUTPUT_DIR, "stage2_v3_str.jsonl")

    with open(out_path, "w") as f:
        for r in valid:
            f.write(json.dumps({
                "prompt": make_prompt(r.get("question", "")),
                "test_cases": get_test_cases(r),
            }, ensure_ascii=False) + "\n")

    with open(out_str_path, "w") as f:
        for r in valid:
            f.write(json.dumps({
                "prompt": make_prompt(r.get("question", "")),
                "test_cases": json.dumps(get_test_cases(r), ensure_ascii=False),
            }, ensure_ascii=False) + "\n")

    # 4. 打印统计
    total = len(competition_records)
    print(f"\n{'='*70}")
    print(f"Competition 数据清洗结果")
    print(f"{'='*70}")
    print(f"  原始数量:     {total}")
    print(f"  保留 (full_pass): {stats['full_pass']} ({stats['full_pass']/total:.1%})")
    print(f"  过滤 (fail):      {stats['fail']} ({stats['fail']/total:.1%})")
    if stats["invalid_test_cases"]:
        print(f"  过滤 (bad tc):    {stats['invalid_test_cases']}")
    print(f"{'='*70}")
    print(f"  输出: {out_path} ({len(valid)} samples)")
    print(f"  输出: {out_str_path} ({len(valid)} samples)")

    # 5. 保存过滤详情
    detail_path = os.path.join(OUTPUT_DIR, "stage2_v3_clean_log.json")
    with open(detail_path, "w") as f:
        json.dump({
            "total": total,
            "valid": len(valid),
            "invalid_count": len(invalid),
            "stats": dict(stats),
            "invalid_details": invalid[:50],
        }, f, indent=2, ensure_ascii=False)
    print(f"  清洗日志: {detail_path}")


if __name__ == "__main__":
    main()
