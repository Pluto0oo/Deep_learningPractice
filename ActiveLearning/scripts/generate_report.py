import argparse
import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.aggregate_results import aggregate_results


def generate_report(df, output_path: str) -> None:
    report = f"""# 实验结果报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 概述

本报告总结了共 {len(df)} 个主动学习实验的结果。

## 实验摘要

| 实验ID | 名称 | 模型 | 随机种子 | 重复次数 |
|--------|------|-------|----------|----------|
"""
    
    for _, row in df.iterrows():
        report += f"| `{row['exp_id']}` | {row['name']} | {row['model_type']} | {row['seed']} | {row['repeat_times']} |\n"
    
    report += "\n## 性能指标\n\n"
    
    accuracy_cols = [col for col in df.columns if "accuracies_mean" in col]
    f1_cols = [col for col in df.columns if "f1_scores_mean" in col]
    
    if accuracy_cols:
        report += "### 采样效率对比（准确率）\n\n"
        report += "| 实验ID | 采样策略 | 10%标注 | 30%标注 | 50%标注 |\n"
        report += "|--------|----------|---------|---------|---------|\n"
        
        for _, row in df.iterrows():
            for col in accuracy_cols:
                strategy = col.replace("_accuracies_mean", "")
                acc_values = row[col]
                if isinstance(acc_values, list) and len(acc_values) >= 3:
                    report += f"| `{row['exp_id']}` | {strategy.capitalize()} | {acc_values[0]:.4f} | {acc_values[1]:.4f} | {acc_values[2]:.4f} |\n"
    
    if f1_cols:
        report += "\n### F1分数\n\n"
        report += "| 实验ID | 采样策略 | 10%标注 | 30%标注 | 50%标注 |\n"
        report += "|--------|----------|---------|---------|---------|\n"
        
        for _, row in df.iterrows():
            for col in f1_cols:
                strategy = col.replace("_f1_scores_mean", "")
                f1_values = row[col]
                if isinstance(f1_values, list) and len(f1_values) >= 3:
                    report += f"| `{row['exp_id']}` | {strategy.capitalize()} | {f1_values[0]:.4f} | {f1_values[1]:.4f} | {f1_values[2]:.4f} |\n"
    
    report += "\n## 最佳性能\n\n"
    
    if accuracy_cols:
        best_acc_col = accuracy_cols[0]
        best_idx = df[best_acc_col].apply(lambda x: x[2] if isinstance(x, list) and len(x) >= 3 else 0).idxmax()
        best_row = df.loc[best_idx]
        best_values = best_row[best_acc_col]
        report += f"- **50%标注时最佳准确率**: `{best_row['exp_id']}` ({best_row['name']}) - {best_values[2]:.4f}\n"
    
    report += "\n## 配置对比\n\n"
    
    report += "### 模型架构\n\n"
    model_types = df["model_type"].unique()
    for model_type in model_types:
        count = len(df[df["model_type"] == model_type])
        report += f"- **{model_type}**: {count} 个实验\n"
    
    report += "\n### 实验设置\n\n"
    report += "- **随机种子**: " + ", ".join(sorted(df["seed"].astype(str).unique())) + "\n"
    report += "- **重复次数**: " + ", ".join(sorted(df["repeat_times"].astype(str).unique())) + "\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description="从所有实验生成综合报告")
    parser.add_argument("--results_dir", type=str, default="results", 
                        help="包含实验结果的目录")
    parser.add_argument("--output", type=str, default="reports/final_report.md", 
                        help="报告输出路径")
    args = parser.parse_args()
    
    df = aggregate_results(args.results_dir)
    
    if df.empty:
        print("未找到结果用于生成报告。")
        return
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_report(df, args.output)
    
    print(f"报告已生成并保存到: {args.output}")


if __name__ == "__main__":
    main()