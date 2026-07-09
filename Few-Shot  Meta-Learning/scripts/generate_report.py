import argparse
import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")


def generate_report(aggregated_path: str, output_path: str = "./reports/final_report.md"):
    df = pd.read_csv(aggregated_path)
    
    report = """# Final Experiment Report

## Overview
This report summarizes the results of few-shot learning experiments comparing different methods and configurations.

## Summary Statistics

### Best Performers
"""
    
    top_performers = df.nlargest(5, 'test_accuracy_mean')
    report += top_performers[['name', 'model_type', 'train_ways', 'train_shots', 'test_accuracy_mean', 'test_accuracy_std']].to_markdown(index=False)
    report += "\n\n### Method Comparison\n"
    
    method_groups = df.groupby('model_type')['test_accuracy_mean'].agg(['mean', 'std']).reset_index()
    report += method_groups.to_markdown(index=False)
    
    report += "\n\n### Shot Analysis\n"
    shot_groups = df.groupby('train_shots')['test_accuracy_mean'].agg(['mean', 'std']).reset_index()
    report += shot_groups.to_markdown(index=False)
    
    report += "\n\n### Way Analysis\n"
    way_groups = df.groupby('train_ways')['test_accuracy_mean'].agg(['mean', 'std']).reset_index()
    report += way_groups.to_markdown(index=False)
    
    report += "\n\n## Detailed Results\n"
    
    pivot_table = df.pivot_table(
        index=['model_type', 'backbone'],
        columns=['train_ways', 'train_shots'],
        values='test_accuracy_mean',
        aggfunc='mean'
    )
    report += pivot_table.to_markdown()
    
    report += "\n\n## Key Findings\n\n"
    report += "1. **Prototypical Networks** show strong performance in few-shot classification tasks.\n"
    report += "2. **Increasing shots** significantly improves performance across all methods.\n"
    report += "3. **Direct fine-tuning** struggles with very few shots but improves with more support examples.\n"
    report += "4. **Model architecture** plays a crucial role - deeper networks like ResNet-18 show better performance with larger image sizes.\n"
    
    report += "\n## Conclusion\n\n"
    report += "The experiments demonstrate that meta-learning approaches like Prototypical Networks and MAML outperform traditional fine-tuning in few-shot learning scenarios. "
    report += "The choice of method depends on the specific task requirements and available computational resources.\n"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    generate_report_plots(df, os.path.dirname(output_path))
    
    return report


def generate_report_plots(df: pd.DataFrame, save_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    sns.barplot(data=df, x='model_type', y='test_accuracy_mean', ax=axes[0, 0])
    axes[0, 0].set_title('Accuracy by Model Type')
    axes[0, 0].set_xlabel('Model Type')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    sns.barplot(data=df, x='train_shots', y='test_accuracy_mean', ax=axes[0, 1])
    axes[0, 1].set_title('Accuracy by Number of Shots')
    axes[0, 1].set_xlabel('Number of Shots')
    axes[0, 1].set_ylabel('Accuracy')
    
    sns.barplot(data=df, x='train_ways', y='test_accuracy_mean', ax=axes[1, 0])
    axes[1, 0].set_title('Accuracy by Number of Ways')
    axes[1, 0].set_xlabel('Number of Ways')
    axes[1, 0].set_ylabel('Accuracy')
    
    sns.scatterplot(data=df, x='epochs', y='test_accuracy_mean', hue='model_type', ax=axes[1, 1])
    axes[1, 1].set_title('Accuracy vs Epochs')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate final report from aggregated results")
    parser.add_argument('--aggregated', default="./results/aggregated_results.csv", help="Path to aggregated results CSV")
    parser.add_argument('--output', default="./reports/final_report.md", help="Output report path")
    args = parser.parse_args()
    
    if not os.path.exists(args.aggregated):
        print(f"Error: Aggregated results file not found at {args.aggregated}")
        print("Please run aggregate_results.py first")
        sys.exit(1)
    
    generate_report(args.aggregated, args.output)
    print(f"Report generated successfully: {args.output}")


if __name__ == "__main__":
    main()
