import argparse
import sys
import os
import json
import pandas as pd
from typing import Dict, List


def aggregate_results(results_dir: str = "./results") -> pd.DataFrame:
    experiments = []
    
    for exp_dir in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        
        config_path = os.path.join(exp_path, 'config_used.yaml')
        metrics_path = os.path.join(exp_path, 'metrics.json')
        
        if not os.path.exists(config_path) or not os.path.exists(metrics_path):
            continue
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            experiment_info = {
                'exp_id': exp_dir,
                'name': config['experiment']['name'],
                'model_type': config['model']['type'],
                'backbone': config['model']['backbone'],
                'train_ways': config['data']['train_ways'],
                'train_shots': config['data']['train_shots'],
                'test_ways': config['data']['test_ways'],
                'test_shots': config['data']['test_shots'],
                'epochs': config['training']['epochs'],
                'meta_lr': config['training']['meta_lr'],
                'repeat_times': config['experiment']['repeat_times'],
            }
            
            if 'test_accuracy' in metrics:
                if isinstance(metrics['test_accuracy'], dict):
                    experiment_info['test_accuracy_mean'] = metrics['test_accuracy']['mean']
                    experiment_info['test_accuracy_std'] = metrics['test_accuracy']['std']
                else:
                    experiment_info['test_accuracy_mean'] = metrics['test_accuracy']
                    experiment_info['test_accuracy_std'] = 0.0
            
            if 'test_loss' in metrics:
                if isinstance(metrics['test_loss'], dict):
                    experiment_info['test_loss_mean'] = metrics['test_loss']['mean']
                else:
                    experiment_info['test_loss_mean'] = metrics['test_loss']
            
            experiments.append(experiment_info)
        
        except Exception as e:
            print(f"Error processing {exp_dir}: {e}")
            continue
    
    df = pd.DataFrame(experiments)
    return df


def main():
    parser = argparse.ArgumentParser(description="Aggregate results from all experiments")
    parser.add_argument('--results_dir', default="./results", help="Directory containing results")
    parser.add_argument('--output', default="./results/aggregated_results.csv", help="Output CSV file path")
    parser.add_argument('--sort_by', default="test_accuracy_mean", help="Column to sort by")
    parser.add_argument('--ascending', action='store_true', help="Sort in ascending order")
    parser.add_argument('--filter', nargs='*', help="Filter conditions (e.g., model_type=protonet)")
    args = parser.parse_args()
    
    df = aggregate_results(args.results_dir)
    
    if args.filter:
        for condition in args.filter:
            key, value = condition.split('=')
            if key in df.columns:
                try:
                    value = float(value)
                except ValueError:
                    pass
                df = df[df[key] == value]
    
    df = df.sort_values(by=args.sort_by, ascending=args.ascending)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    
    print(f"Aggregated {len(df)} experiments")
    print(f"Results saved to {args.output}")
    print("\nSummary:")
    print(df[['exp_id', 'name', 'model_type', 'train_ways', 'train_shots', 'test_accuracy_mean', 'test_accuracy_std']].to_string(index=False))


if __name__ == "__main__":
    main()
