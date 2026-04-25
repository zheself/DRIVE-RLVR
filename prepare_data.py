import json
import random
import os
from collections import defaultdict

INPUT_FILE = "/mnt/sdc/ubuntu/cjz_projects/datasets/codeparrot/apps/train.jsonl"
OUTPUT_DIR = "/mnt/sdc/ubuntu/cjz_projects/DRIVE/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load all records
records = []
with open(INPUT_FILE) as f:
    for line in f:
        records.append(json.loads(line))

# Audit before filtering
before = defaultdict(int)
for r in records:
    before[r.get("difficulty", "unknown")] += 1
print("=== Before filtering ===")
for k, v in sorted(before.items()):
    print(f"  {k}: {v}")

# Filter
def valid(r):
    try:
        io = json.loads(r.get("input_output", ""))
        ins, outs = io.get("inputs", []), io.get("outputs", [])
        return len(ins) == len(outs) and len(ins) > 0
    except Exception:
        return False

filtered = [r for r in records if valid(r)]

# Audit after filtering
after = defaultdict(int)
for r in filtered:
    after[r.get("difficulty", "unknown")] += 1
print("\n=== After filtering ===")
for k, v in sorted(after.items()):
    removed = before[k] - v
    pct = removed / before[k] * 100 if before[k] else 0
    print(f"  {k}: {v} (removed {removed}, {pct:.1f}%)")

def get_test_cases(r):
    io = json.loads(r["input_output"])
    ins, outs = io["inputs"], io["outputs"]
    fn_name = io.get("fn_name", None)
    cases = []
    for i in range(min(5, len(ins))):
        inp = ins[i]
        out = outs[i]
        # stdin 型中部分 input/output 是 list（APPS 格式不一致），归一化为字符串
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

def export(samples, path, n=None):
    random.seed(42)
    if n is not None:
        chosen = random.sample(samples, min(n, len(samples)))
    else:
        chosen = samples[:]
        random.shuffle(chosen)
    with open(path, "w") as f:
        for r in chosen:
            f.write(json.dumps({
                "prompt": make_prompt(r.get("question", "")),
                "test_cases": get_test_cases(r)
            }, ensure_ascii=False) + "\n")
    print(f"\nExported {len(chosen)} records -> {path}")


def export_str(samples, path, n=None):
    """导出 test_cases 为 JSON 字符串格式（OpenRLHF label 需要字符串）"""
    random.seed(42)
    if n is not None:
        chosen = random.sample(samples, min(n, len(samples)))
    else:
        chosen = samples[:]
        random.shuffle(chosen)
    with open(path, "w") as f:
        for r in chosen:
            f.write(json.dumps({
                "prompt": make_prompt(r.get("question", "")),
                "test_cases": json.dumps(get_test_cases(r), ensure_ascii=False)
            }, ensure_ascii=False) + "\n")
    print(f"\nExported {len(chosen)} records (str format) -> {path}")


def has_fn_name(r):
    try:
        io = json.loads(r["input_output"])
        return bool(io.get("fn_name"))
    except:
        return False


by_diff = defaultdict(list)
for r in filtered:
    by_diff[r.get("difficulty", "unknown")].append(r)

# === Stage1 v2 数据：全部 introductory + interview fn_name 型 ===
stage1_v2_samples = by_diff["introductory"] + [r for r in by_diff["interview"] if has_fn_name(r)]
export(stage1_v2_samples, f"{OUTPUT_DIR}/stage1_v2.jsonl")
export_str(stage1_v2_samples, f"{OUTPUT_DIR}/stage1_v2_str.jsonl")

# === Stage2 v2 数据：全部 competition ===
export(by_diff["competition"], f"{OUTPUT_DIR}/stage2_v2.jsonl")
export_str(by_diff["competition"], f"{OUTPUT_DIR}/stage2_v2_str.jsonl")

# === 测试集：从 test.jsonl 导出 ===
print("\n" + "="*70)
print("Processing test.jsonl for held-out test sets...")
print("="*70)

TEST_FILE = "/mnt/sdc/ubuntu/cjz_projects/datasets/codeparrot/apps/test.jsonl"
test_records = []
with open(TEST_FILE) as f:
    for line in f:
        test_records.append(json.loads(line))

# 过滤有效的测试数据
test_filtered = [r for r in test_records if valid(r)]

# 按难度分组
test_by_diff = defaultdict(list)
for r in test_filtered:
    test_by_diff[r.get("difficulty", "unknown")].append(r)

print("\n=== test.jsonl 数据分布 ===")
for k in sorted(test_by_diff.keys()):
    fn_count = sum(1 for r in test_by_diff[k] if has_fn_name(r))
    stdin_count = len(test_by_diff[k]) - fn_count
    print(f"  {k}: {len(test_by_diff[k])} total ({fn_count} fn_name, {stdin_count} stdin)")

# 导出 introductory 测试集（用于评测 stage1 模型）
export(test_by_diff["introductory"], f"{OUTPUT_DIR}/test_introductory.jsonl")
export_str(test_by_diff["introductory"], f"{OUTPUT_DIR}/test_introductory_str.jsonl")

# 导出 competition 测试集（用于评测 stage2 模型）
export(test_by_diff["competition"], f"{OUTPUT_DIR}/test_competition.jsonl")
export_str(test_by_diff["competition"], f"{OUTPUT_DIR}/test_competition_str.jsonl")

# 导出 interview 测试集（可选，用于中等难度评测）
export(test_by_diff["interview"], f"{OUTPUT_DIR}/test_interview.jsonl")
export_str(test_by_diff["interview"], f"{OUTPUT_DIR}/test_interview_str.jsonl")

print("\n" + "="*70)
print("测试集说明:")
print("  - test_introductory: 用于评测 stage1 模型的泛化能力（与训练难度相同但题目不同）")
print("  - test_competition: 用于评测 stage2 模型的泛化能力")
print("  - test_interview: 可选的中等难度测试集")
print("  注意: test.jsonl 中大部分是 stdin 型，与训练数据（fn_name 型为主）类型不同")
print("="*70)

print("\nDone.")
