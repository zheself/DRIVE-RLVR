#!/usr/bin/env python3
"""
生成模型对比报告
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


def load_results(results_dir: str) -> Dict[str, Dict]:
    """加载所有评估结果"""
    results = {}
    results_path = Path(results_dir)

    for result_file in results_path.glob("*_results.json"):
        model_name = result_file.stem.replace("_results", "")
        with open(result_file, 'r', encoding='utf-8') as f:
            results[model_name] = json.load(f)

    return results


def generate_markdown_report(results: Dict[str, Dict], output_file: str):
    """生成 Markdown 格式的对比报告"""

    lines = [
        "# DRIVE Stage1 模型评估对比报告",
        "",
        f"**评估时间**: {Path(output_file).parent.name}",
        "",
        "---",
        "",
        "## 一、整体性能对比",
        "",
        "### 1.1 核心指标",
        "",
        "| 模型 | 样本数 | 平均分数 | 截断率 | 平均响应长度 |",
        "|---|---|---|---|---|"
    ]

    # 按模型名称排序（baseline 放第一个）
    model_order = ['baseline', 'v2', 'v3']
    sorted_models = sorted(results.keys(), key=lambda x: model_order.index(x) if x in model_order else 999)

    for model_name in sorted_models:
        data = results[model_name]
        summary = data['summary']

        lines.append(
            f"| {model_name} | "
            f"{summary['num_samples']} | "
            f"{summary['avg_score']:.4f} | "
            f"{summary['truncated_rate']:.1%} | "
            f"{summary['avg_response_length']:.1f} chars |"
        )

    lines.extend([
        "",
        "**指标说明**：",
        "- **平均分数**：模型在测试用例上的平均得分（0-1之间，越高越好）",
        "- **截断率**：生成的代码被截断的样本比例",
        "- **平均响应长度**：模型生成的平均字符数",
        "",
        "---",
        "",
        "## 二、相对提升分析",
        ""
    ])

    # 计算相对于 baseline 的提升
    if 'baseline' in results:
        baseline_summary = results['baseline']['summary']

        lines.extend([
            "### 2.1 相对于原始模型的提升",
            "",
            "| 模型 | 分数提升 | 截断率变化 | 响应长度变化 |",
            "|---|---|---|---|"
        ])

        for model_name in sorted_models:
            if model_name == 'baseline':
                continue

            summary = results[model_name]['summary']

            score_delta = summary['avg_score'] - baseline_summary['avg_score']
            truncated_delta = summary['truncated_rate'] - baseline_summary['truncated_rate']
            length_delta = summary['avg_response_length'] - baseline_summary['avg_response_length']

            lines.append(
                f"| {model_name} | "
                f"{score_delta:+.4f} ({score_delta/baseline_summary['avg_score']*100:+.1f}%) | "
                f"{truncated_delta:+.1%} | "
                f"{length_delta:+.1f} chars |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 三、详细分析",
        ""
    ])

    # 为每个模型生成详细分析
    for model_name in sorted_models:
        data = results[model_name]
        summary = data['summary']
        detailed_results = data['results']

        lines.extend([
            f"### 3.{sorted_models.index(model_name) + 1} {model_name}",
            "",
            f"**模型路径**: `{summary['model_path']}`",
            "",
            "#### 性能统计",
            "",
            f"- 评估样本数: {summary['num_samples']}",
            f"- 成功评估: {summary['successful_evaluations']}",
            f"- 失败评估: {summary['failed_evaluations']}",
            f"- 平均分数: {summary['avg_score']:.4f}",
            f"- 截断样本数: {summary['truncated_count']}",
            f"- 截断率: {summary['truncated_rate']:.1%}",
            f"- 平均响应长度: {summary['avg_response_length']:.1f} chars",
            ""
        ])

    lines.extend([
        "---",
        "",
        "## 四、结论与建议",
        ""
    ])

    # 找出最佳模型
    best_model = max(results.keys(), key=lambda x: results[x]['summary']['avg_score'])
    best_score = results[best_model]['summary']['avg_score']

    lines.extend([
        f"### 最佳模型: {best_model}",
        "",
        f"- 平均分数: {best_score:.4f}",
        f"- 截断率: {results[best_model]['summary']['truncated_rate']:.1%}",
        f"- 平均响应长度: {results[best_model]['summary']['avg_response_length']:.1f} chars",
        ""
    ])

    if 'baseline' in results and best_model != 'baseline':
        improvement = best_score - results['baseline']['summary']['avg_score']
        improvement_pct = improvement / results['baseline']['summary']['avg_score'] * 100
        lines.extend([
            f"### 训练效果",
            "",
            f"相对于原始模型，{best_model} 的平均分数提升了 **{improvement:+.4f}** (**{improvement_pct:+.1f}%**)",
            ""
        ])

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Generate comparison report')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing result JSON files')
    parser.add_argument('--output', type=str, required=True, help='Output markdown file')

    args = parser.parse_args()

    # 加载结果
    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)

    if not results:
        print("No results found!")
        return

    print(f"Found {len(results)} model results: {', '.join(results.keys())}")

    # 生成报告
    print(f"Generating comparison report...")
    generate_markdown_report(results, args.output)

    print(f"Report saved to: {args.output}")


if __name__ == '__main__':
    main()
