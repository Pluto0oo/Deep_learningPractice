import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, generate_exp_id, save_config
from src.logger import setup_logger
from src.data_loader import get_omniglot_dataset, create_task_loader
from src.models import create_model
from src.trainer import train_model, evaluate_model
from src.results import setup_results_dir, save_experiment_results, save_repeat_results, save_checkpoint
from src.utils import set_seed, get_device, count_parameters
from src.visualization import plot_comparison
import json


def run_single_config(config_path, exp_id_prefix):
    config = load_config(config_path)
    config['experiment']['exp_id'] = f"{exp_id_prefix}_{config['experiment']['name']}"
    
    logger = setup_logger(config['experiment']['exp_id'], config['logging']['log_dir'], config['logging']['level'])
    logger.info(f"Running comparison experiment: {config['experiment']['name']}")
    
    device = get_device(config)
    
    exp_dir = setup_results_dir(config['experiment']['exp_id'], config['results']['save_dir'])
    save_config(config, os.path.join(exp_dir, 'config_used.yaml'))
    
    repeat_times = config['experiment']['repeat_times']
    repeat_results = []
    
    for repeat in range(repeat_times):
        logger.info(f"Starting repeat {repeat + 1}/{repeat_times}")
        set_seed(config['experiment']['seed'] + repeat)
        
        if repeat_times > 1:
            repeat_dir = os.path.join(exp_dir, 'repeats', f'repeat_{repeat + 1:03d}')
            os.makedirs(repeat_dir, exist_ok=True)
        else:
            repeat_dir = exp_dir
        
        train_dataset, test_dataset = get_omniglot_dataset(config)
        train_loader = create_task_loader(train_dataset, config, mode="train")
        test_loader = create_task_loader(test_dataset, config, mode="test")
        
        model = create_model(config)
        model = model.to(device)
        
        metrics_history = train_model(model, train_loader, config, device, logger)
        test_loss, test_acc = evaluate_model(model, test_loader, config, device)
        
        if config['results'].get('save_checkpoints', True):
            save_checkpoint(model, os.path.join(repeat_dir, 'checkpoints'))
        
        final_metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
        }
        repeat_results.append(final_metrics)
        
        if repeat_times == 1:
            save_experiment_results(config, metrics_history, final_metrics, exp_dir)
    
    if repeat_times > 1:
        aggregated = save_repeat_results(repeat_results, exp_dir)
        save_experiment_results(config, [], aggregated, exp_dir)
        return aggregated
    else:
        return repeat_results[0]


def run_comparison(config_paths, exp_id):
    results = {}
    
    for config_path in config_paths:
        config = load_config(config_path)
        result = run_single_config(config_path, exp_id)
        results[config['experiment']['name']] = result
    
    comparison_dir = os.path.join('./results', exp_id)
    os.makedirs(comparison_dir, exist_ok=True)
    
    plot_comparison(results, comparison_dir)
    
    with open(os.path.join(comparison_dir, 'comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    generate_comparison_report(results, comparison_dir)
    
    return results


def generate_comparison_report(results, save_path):
    report = """# Comparison Report

## Methods Compared
"""
    
    for method, metrics in results.items():
        report += f"- {method}\n"
    
    report += "\n## Results Summary\n\n"
    report += "| Method | Test Accuracy | Test Loss |\n"
    report += "|--------|---------------|-----------|\n"
    
    for method, metrics in results.items():
        if 'test_accuracy' in metrics:
            acc = metrics['test_accuracy']
            if isinstance(acc, dict):
                acc_val = f"{acc['mean']:.4f} (±{acc['std']:.4f})"
            else:
                acc_val = f"{acc:.4f}"
            
            loss = metrics.get('test_loss', 'N/A')
            if isinstance(loss, dict):
                loss_val = f"{loss['mean']:.4f}"
            else:
                loss_val = f"{loss:.4f}"
            
            report += f"| {method} | {acc_val} | {loss_val} |\n"
    
    with open(os.path.join(save_path, 'comparison_report.md'), 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description="Run comparison between multiple experiments")
    parser.add_argument('--configs', nargs='+', required=True, help="Paths to configuration files")
    parser.add_argument('--exp_id', default=None, help="Comparison experiment ID")
    args = parser.parse_args()
    
    exp_id = args.exp_id if args.exp_id else generate_exp_id()
    
    logger = setup_logger(exp_id, './logs', 'INFO')
    logger.info(f"Starting comparison experiment: {exp_id}")
    logger.info(f"Comparing {len(args.configs)} configurations")
    
    run_comparison(args.configs, exp_id)
    
    logger.info("Comparison completed successfully")


if __name__ == "__main__":
    main()
