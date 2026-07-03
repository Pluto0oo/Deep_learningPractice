import argparse
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.aggregate_results import aggregate_results


def plot_accuracy_comparison(df, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    accuracy_cols = [col for col in df.columns if "test_accuracy" in col or "accuracy_mean" in col]
    if not accuracy_cols:
        print("No accuracy columns found")
        return
    
    plt.figure(figsize=(12, 6))
    
    sns.barplot(data=df, x="exp_id", y=accuracy_cols[0], palette="viridis")
    plt.title("Test Accuracy Comparison")
    plt.xlabel("Experiment ID")
    plt.ylabel("Test Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=300)
    plt.close()


def plot_metric_distribution(df, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    metric_cols = [col for col in df.columns if col.startswith("test_") and col != "test_accuracy"]
    
    if not metric_cols:
        print("No test metric columns found")
        return
    
    n_cols = 2
    n_rows = (len(metric_cols) + 1) // n_cols
    
    plt.figure(figsize=(10, 4 * n_rows))
    
    for i, metric in enumerate(metric_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df[metric], kde=True)
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_distributions.png"), dpi=300)
    plt.close()


def analyze_best_performers(df) -> None:
    print("\n=== Best Performers Analysis ===")
    
    accuracy_cols = [col for col in df.columns if "test_accuracy" in col or "accuracy_mean" in col]
    if accuracy_cols:
        best_acc = df.loc[df[accuracy_cols[0]].idxmax()]
        print(f"\nBest Test Accuracy:")
        print(f"  Exp ID: {best_acc['exp_id']}")
        print(f"  Name: {best_acc['name']}")
        print(f"  Model: {best_acc['model_type']}")
        print(f"  Accuracy: {best_acc[accuracy_cols[0]]:.4f}")
    
    f1_cols = [col for col in df.columns if "test_f1_macro" in col or "f1_macro_mean" in col]
    if f1_cols:
        best_f1 = df.loc[df[f1_cols[0]].idxmax()]
        print(f"\nBest F1 Score:")
        print(f"  Exp ID: {best_f1['exp_id']}")
        print(f"  Name: {best_f1['name']}")
        print(f"  Model: {best_f1['model_type']}")
        print(f"  F1 Score: {best_f1[f1_cols[0]]:.4f}")


def analyze_correlations(df) -> None:
    print("\n=== Correlation Analysis ===")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation analysis")
        return
    
    corr_matrix = df[numeric_cols].corr()
    
    print("\nCorrelation with test_accuracy:")
    accuracy_cols = [col for col in corr_matrix.columns if "test_accuracy" in col or "accuracy_mean" in col]
    if accuracy_cols:
        acc_corr = corr_matrix[accuracy_cols[0]].sort_values(ascending=False)
        for col, corr in acc_corr.items():
            if col != accuracy_cols[0]:
                print(f"  {col}: {corr:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize experiment results")
    parser.add_argument("--results_dir", type=str, default="results", 
                        help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default="reports/plots", 
                        help="Output directory for plots")
    args = parser.parse_args()
    
    df = aggregate_results(args.results_dir)
    
    if df.empty:
        print("No results found to analyze.")
        return
    
    print(f"Analyzing {len(df)} experiments...")
    
    plot_accuracy_comparison(df, args.output_dir)
    plot_metric_distribution(df, args.output_dir)
    
    analyze_best_performers(df)
    analyze_correlations(df)
    
    print(f"\nPlots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()