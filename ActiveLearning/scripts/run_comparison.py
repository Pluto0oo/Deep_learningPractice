import argparse
import os
import sys
import json
from typing import List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils import generate_exp_id


def run_experiment(config_path: str, exp_id: str = None) -> str:
    import subprocess
    
    cmd = [sys.executable, "scripts/run_experiment.py", "--config", config_path]
    if exp_id:
        cmd.extend(["--exp_id", exp_id])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running {config_path}:")
        print(result.stderr)
        return None
    
    exp_id_line = [line for line in result.stdout.split("\n") if "Starting experiment:" in line]
    if exp_id_line:
        return exp_id_line[0].split(":")[-1].strip()
    
    return exp_id


def generate_comparison_report(exp_ids: List[str], config_paths: List[str], 
                               output_path: str) -> None:
    report = f"""# Experiment Comparison Report

Generated on: {generate_exp_id().split('_')[1]}

## Experiments Compared

| Experiment | Config | Exp ID |
|------------|--------|--------|
"""
    
    for i, (config_path, exp_id) in enumerate(zip(config_paths, exp_ids)):
        report += f"| {i+1} | `{os.path.basename(config_path)}` | `{exp_id}` |\n"
    
    report += "\n## Results Summary\n\n"
    
    all_metrics = {}
    for exp_id in exp_ids:
        metrics_path = os.path.join("results", exp_id, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            all_metrics[exp_id] = metrics
    
    if all_metrics:
        report += "### Test Accuracy Comparison\n\n"
        report += "| Experiment ID | Test Accuracy |\n"
        report += "|---------------|---------------|\n"
        
        for exp_id, metrics in all_metrics.items():
            test_acc = metrics.get("test_accuracy", metrics.get("accuracy_mean", "N/A"))
            if isinstance(test_acc, float):
                test_acc = f"{test_acc:.4f}"
            report += f"| `{exp_id}` | {test_acc} |\n"
    
    report += "\n## Detailed Metrics\n\n"
    
    for exp_id, metrics in all_metrics.items():
        report += f"### {exp_id}\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        
        for key, value in sorted(metrics.items()):
            if isinstance(value, (int, float)):
                report += f"| {key} | {value:.4f} |\n"
            else:
                report += f"| {key} | {value} |\n"
        report += "\n"
    
    with open(output_path, "w") as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description="Run multiple experiments and compare results")
    parser.add_argument("--configs", type=str, nargs="+", required=True, 
                        help="Paths to config files to compare")
    parser.add_argument("--output", type=str, default="reports/comparison_report.md", 
                        help="Output path for comparison report")
    args = parser.parse_args()
    
    print(f"Running comparison for {len(args.configs)} experiments...")
    
    exp_ids = []
    for i, config_path in enumerate(args.configs):
        print(f"\n[{i+1}/{len(args.configs)}] Running: {os.path.basename(config_path)}")
        
        exp_id = run_experiment(config_path)
        if exp_id:
            exp_ids.append(exp_id)
            print(f"   Experiment ID: {exp_id}")
        else:
            print(f"   Failed to run experiment")
    
    if exp_ids:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        generate_comparison_report(exp_ids, args.configs, args.output)
        print(f"\nComparison report saved to: {args.output}")
    else:
        print("\nNo experiments completed successfully.")


if __name__ == "__main__":
    main()