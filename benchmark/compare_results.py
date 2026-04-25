#!/usr/bin/env python3
"""
compare_results.py — 汇总 lm_eval 评测结果，生成对比表格和报告

用法:
    python compare_results.py [--results_dir ./results] [--output report.md]
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime


# 关注的指标（任务名 -> 指标名）
TASK_METRICS = {
    "humaneval": "pass@1,create_test",
    "mbpp": "pass_at_1,none",
    "gsm8k": "exact_match,strict-match",
    "mmlu": "acc,none",
    "hellaswag": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "winogrande": "acc,none",
    "truthfulqa_mc2": "acc,none",
}

# 任务分类
TASK_CATEGORIES = {
    "代码生成": ["humaneval", "mbpp"],
    "数学推理": ["gsm8k"],
    "常识推理": ["hellaswag", "arc_challenge", "winogrande"],
    "通用知识": ["mmlu", "truthfulqa_mc2"],
}

# 模型显示名
MODEL_DISPLAY = {
    "baseline": "Baseline",
    "stage1_v5": "Stage1 v5",
    "stage2_r64": "Stage2 r64",
    "stage2_v2_r16": "Stage2 v2 r16",
    "stage2_v2_r64": "Stage2 v2 r64",
}


def find_results_file(model_dir: Path) -> dict | None:
    """在模型结果目录中查找 lm_eval 的 results JSON"""
    # lm_eval 输出格式: results/<model_name>/<timestamp>/results.json
    # 或直接在 model_dir 下
    for pattern in ["**/results*.json", "results*.json"]:
        files = list(model_dir.glob(pattern))
        if files:
            # 取最新的
            latest = max(files, key=lambda f: f.stat().st_mtime)
            with open(latest) as f:
                return json.load(f)
    return None


def extract_scores(results: dict) -> dict:
    """从 lm_eval results 中提取各任务分数"""
    scores = {}
    if "results" not in results:
        return scores

    for task_name, metric_key in TASK_METRICS.items():
        task_results = results["results"].get(task_name, {})
        if not task_results:
            continue

        # lm_eval 的指标格式可能是 "acc,none" 或 "pass@1" 等
        score = None
        # 尝试精确匹配
        if metric_key in task_results:
            score = task_results[metric_key]
        else:
            # 尝试模糊匹配
            for k, v in task_results.items():
                if metric_key.split(",")[0] in k:
                    score = v
                    break

        if score is not None:
            scores[task_name] = round(float(score) * 100, 2) if float(score) <= 1.0 else round(float(score), 2)

    return scores


def generate_report(all_scores: dict, output_path: str):
    """生成 Markdown 对比报告"""
    models = list(all_scores.keys())
    lines = []

    lines.append("# lm_eval 通用 Benchmark 对比报告")
    lines.append("")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 总览表
    lines.append("## 一、总览")
    lines.append("")

    # 表头
    header = "| 类别 | 任务 |"
    sep = "|------|------|"
    for m in models:
        display = MODEL_DISPLAY.get(m, m)
        header += f" {display} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    # 按类别输出
    for category, tasks in TASK_CATEGORIES.items():
        for i, task in enumerate(tasks):
            cat_label = category if i == 0 else ""
            row = f"| {cat_label} | {task} |"
            for m in models:
                score = all_scores[m].get(task, "—")
                if isinstance(score, (int, float)):
                    row += f" {score:.2f} |"
                else:
                    row += f" {score} |"
            lines.append(row)

    lines.append("")

    # 分类别平均分
    lines.append("## 二、分类别平均分")
    lines.append("")
    header = "| 类别 |"
    sep = "|------|"
    for m in models:
        header += f" {MODEL_DISPLAY.get(m, m)} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    for category, tasks in TASK_CATEGORIES.items():
        row = f"| {category} |"
        for m in models:
            task_scores = [all_scores[m][t] for t in tasks if t in all_scores[m]]
            if task_scores:
                avg = sum(task_scores) / len(task_scores)
                row += f" {avg:.2f} |"
            else:
                row += " — |"
        lines.append(row)

    lines.append("")

    # 变化分析
    if "baseline" in all_scores and len(models) > 1:
        lines.append("## 三、相对 Baseline 的变化")
        lines.append("")
        header = "| 任务 |"
        sep = "|------|"
        for m in models:
            if m == "baseline":
                continue
            header += f" {MODEL_DISPLAY.get(m, m)} |"
            sep += "------|"
        lines.append(header)
        lines.append(sep)

        for category, tasks in TASK_CATEGORIES.items():
            for task in tasks:
                base_score = all_scores.get("baseline", {}).get(task)
                if base_score is None:
                    continue
                row = f"| {task} |"
                for m in models:
                    if m == "baseline":
                        continue
                    score = all_scores[m].get(task)
                    if score is not None:
                        diff = score - base_score
                        sign = "+" if diff > 0 else ""
                        row += f" {sign}{diff:.2f} |"
                    else:
                        row += " — |"
                lines.append(row)

    lines.append("")
    lines.append("## 四、结论")
    lines.append("")
    lines.append("（请根据数据填写分析结论）")
    lines.append("")

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\n报告已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results",
                        help="lm_eval 结果目录")
    parser.add_argument("--output", default="./benchmark_report.md",
                        help="输出报告路径")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"结果目录不存在: {results_dir}")
        return

    all_scores = {}
    # 按固定顺序处理
    model_order = ["baseline", "stage1_v5", "stage2_r64", "stage2_v2_r16", "stage2_v2_r64"]

    for model_name in model_order:
        model_dir = results_dir / model_name
        if not model_dir.exists():
            print(f"跳过 {model_name}: 目录不存在")
            continue

        results = find_results_file(model_dir)
        if results is None:
            print(f"跳过 {model_name}: 未找到 results.json")
            continue

        scores = extract_scores(results)
        if scores:
            all_scores[model_name] = scores
            print(f"已加载 {model_name}: {len(scores)} 个任务")
        else:
            print(f"跳过 {model_name}: 无有效分数")

    if not all_scores:
        print("没有找到任何评测结果，请先运行 run_all_models.sh")
        return

    generate_report(all_scores, args.output)


if __name__ == "__main__":
    main()
