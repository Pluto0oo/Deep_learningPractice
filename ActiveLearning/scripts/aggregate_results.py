import argparse
import os
import sys
import json
import pandas as pd
from typing import Dict, Any, List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def aggregate_results(results_dir: str = "results") -> pd.DataFrame:
    rows = []
    
    if not os.path.exists(results_dir):
        print(f"结果目录不存在: {results_dir}")
        return pd.DataFrame()
    
    exp_dirs = sorted([d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))])
    
    for exp_id in exp_dirs:
        exp_path = os.path.join(results_dir, exp_id)
        metrics_path = os.path.join(exp_path, "metrics.json")
        config_path = os.path.join(exp_path, "config_used.yaml")
        
        if not os.path.exists(metrics_path):
            continue
        
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        
        config = {}
        if os.path.exists(config_path):
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        
        row = {"exp_id": exp_id}
        
        row.update({
            "name": config.get("experiment", {}).get("name", "未知"),
            "seed": config.get("experiment", {}).get("seed", "未知"),
            "repeat_times": config.get("experiment", {}).get("repeat_times", 1),
            "model_type": config.get("model", {}).get("type", "未知"),
            "epochs": config.get("training", {}).get("epochs", "未知"),
            "batch_size": config.get("training", {}).get("batch_size", "未知"),
            "optimizer": config.get("training", {}).get("optimizer", "未知"),
            "lr": config.get("training", {}).get("learning_rate", "未知"),
        })
        
        for strategy, strategy_metrics in metrics.items():
            if isinstance(strategy_metrics, dict):
                for metric_name, metric_data in strategy_metrics.items():
                    if isinstance(metric_data, dict) and "mean" in metric_data:
                        row[f"{strategy}_{metric_name}_mean"] = metric_data["mean"]
                        row[f"{strategy}_{metric_name}_std"] = metric_data["std"]
        
        rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        return df
    else:
        return pd.DataFrame()


def save_aggregated_results(df: pd.DataFrame, output_dir: str = "reports") -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "aggregated_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"汇总结果已保存到: {csv_path}")
    
    excel_path = os.path.join(output_dir, "aggregated_results.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"汇总结果已保存到: {excel_path}")


def main():
    parser = argparse.ArgumentParser(description="汇总所有实验结果")
    parser.add_argument("--results_dir", type=str, default="results", 
                        help="包含实验结果的目录")
    parser.add_argument("--output_dir", type=str, default="reports", 
                        help="汇总结果输出目录")
    args = parser.parse_args()
    
    df = aggregate_results(args.results_dir)
    
    if df.empty:
        print("未找到结果用于汇总。")
        return
    
    print(f"找到 {len(df)} 个实验。")
    print("\n摘要:")
    print(df[["exp_id", "name", "model_type"]].to_string(index=False))
    
    save_aggregated_results(df, args.output_dir)


if __name__ == "__main__":
    main()