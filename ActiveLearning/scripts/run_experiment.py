import argparse
import os
import sys
import json
import numpy as np
from typing import Dict, Any, List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils import (
    set_seed,
    setup_logger,
    load_config,
    save_config,
    generate_exp_id,
    create_experiment_dirs,
    get_device,
    save_metrics,
)
from src.data_loader import load_data, get_data_stats
from src.trainer import train_model
from src.evaluator import evaluate_model, compute_metrics, predict_model
from src.model import build_model, get_loss_function


def run_single_repeat(config: Dict[str, Any], repeat_idx: int, exp_id: str, 
                      logger, base_dir: str) -> Dict[str, Any]:
    repeat_dir = os.path.join(base_dir, "repeats", f"repeat_{repeat_idx:03d}")
    os.makedirs(repeat_dir, exist_ok=True)
    
    seed = config["experiment"]["seed"] + repeat_idx
    config["experiment"]["seed"] = seed
    
    set_seed(seed)
    logger.info(f"Running repeat {repeat_idx} with seed {seed}")
    
    exp_dirs = {
        "base": repeat_dir,
        "plots": os.path.join(repeat_dir, "plots"),
        "checkpoints": os.path.join(repeat_dir, "checkpoints"),
        "repeats": os.path.join(repeat_dir, "repeats"),
    }
    for dir_path in exp_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    save_config(config, os.path.join(repeat_dir, "config_used.yaml"))
    
    try:
        train_data, test_data = load_data(config)
    except FileNotFoundError:
        from src.data_loader import create_sample_data
        create_sample_data("data/processed")
        train_data, test_data = load_data(config)
    
    logger.info(f"Data stats: {get_data_stats(train_data)}")
    
    device = get_device()
    logger.info(f"Using device: {device}")
    
    final_metrics = train_model(config, train_data, device, logger, exp_dirs)
    
    model = build_model(config).to(device)
    checkpoint_path = os.path.join(exp_dirs["checkpoints"], "best_model.pt")
    if os.path.exists(checkpoint_path):
        model.load_checkpoint(checkpoint_path)
    
    y_pred = predict_model(model, test_data["X"], device)
    test_metrics = compute_metrics(test_data["y"], y_pred)
    
    final_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
    
    save_metrics(final_metrics, os.path.join(repeat_dir, "metrics.json"))
    
    logger.info(f"Repeat {repeat_idx} completed. Test accuracy: {test_metrics['accuracy']:.4f}")
    
    return final_metrics


def compute_repeat_stats(repeat_metrics: List[Dict[str, Any]], stats_dir: str) -> Dict[str, Any]:
    stats = {}
    
    for key in repeat_metrics[0].keys():
        values = [m[key] for m in repeat_metrics]
        stats[f"{key}_mean"] = float(np.mean(values))
        stats[f"{key}_std"] = float(np.std(values))
        stats[f"{key}_min"] = float(np.min(values))
        stats[f"{key}_max"] = float(np.max(values))
    
    os.makedirs(stats_dir, exist_ok=True)
    with open(os.path.join(stats_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    return stats


def generate_summary(config: Dict[str, Any], metrics: Dict[str, Any], exp_dir: str) -> str:
    summary = f"""# Experiment Summary: {config['experiment']['name']}

## Configuration
- Experiment ID: {os.path.basename(exp_dir)}
- Seed: {config['experiment']['seed']}
- Repeat Times: {config['experiment']['repeat_times']}

## Model
- Type: {config['model']['type']}
- Hidden Dimension: {config['model'].get('hidden_dim', 128)}
- Number of Layers: {config['model'].get('num_layers', 2)}

## Training
- Epochs: {config['training'].get('epochs', 100)}
- Batch Size: {config['training'].get('batch_size', 32)}
- Optimizer: {config['training'].get('optimizer', 'adam')}
- Learning Rate: {config['training'].get('learning_rate', 0.001)}

## Results

"""
    
    if "accuracy_mean" in metrics:
        summary += "### Statistics (across repeats)\n\n"
        summary += "| Metric | Mean | Std | Min | Max |\n"
        summary += "|--------|------|-----|-----|-----|\n"
        
        base_metrics = ["train_acc", "val_acc", "test_accuracy"]
        for metric in base_metrics:
            if f"{metric}_mean" in metrics:
                summary += f"| {metric} | {metrics[f'{metric}_mean']:.4f} | {metrics[f'{metric}_std']:.4f} | {metrics[f'{metric}_min']:.4f} | {metrics[f'{metric}_max']:.4f} |\n"
    else:
        summary += "### Final Metrics\n\n"
        summary += "| Metric | Value |\n"
        summary += "|--------|-------|\n"
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                summary += f"| {key} | {value:.4f} |\n"
    
    summary_path = os.path.join(exp_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write(summary)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--exp_id", type=str, default=None, help="Experiment ID (auto-generated if not provided)")
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    exp_id = args.exp_id or generate_exp_id(config["experiment"].get("name", "exp"))
    config["experiment"]["exp_id"] = exp_id
    
    logger = setup_logger(exp_id)
    logger.info(f"Starting experiment: {exp_id}")
    logger.info(f"Config loaded from: {args.config}")
    
    exp_dirs = create_experiment_dirs(exp_id)
    save_config(config, os.path.join(exp_dirs["base"], "config_used.yaml"))
    
    repeat_times = config["experiment"].get("repeat_times", 1)
    
    if repeat_times > 1:
        logger.info(f"Running {repeat_times} repeats...")
        all_repeat_metrics = []
        
        for i in range(repeat_times):
            metrics = run_single_repeat(config, i, exp_id, logger, exp_dirs["base"])
            all_repeat_metrics.append(metrics)
        
        stats_dir = os.path.join(exp_dirs["base"], "repeats", "stats")
        final_metrics = compute_repeat_stats(all_repeat_metrics, stats_dir)
    else:
        final_metrics = run_single_repeat(config, 0, exp_id, logger, exp_dirs["base"])
    
    generate_summary(config, final_metrics, exp_dirs["base"])
    save_metrics(final_metrics, os.path.join(exp_dirs["base"], "metrics.json"))
    
    logger.info(f"Experiment {exp_id} completed successfully!")
    logger.info(f"Results saved to: {exp_dirs['base']}")


if __name__ == "__main__":
    main()