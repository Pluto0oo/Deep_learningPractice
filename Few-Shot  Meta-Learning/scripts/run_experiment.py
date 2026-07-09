import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, generate_exp_id, save_config
from src.logger import setup_logger
from src.data_loader import get_omniglot_dataset, create_task_loader
from src.models import create_model
from src.trainer import train_model, evaluate_model
from src.results import (
    setup_results_dir,
    save_experiment_results,
    save_repeat_results,
    save_checkpoint,
)
from src.utils import set_seed, get_device, count_parameters


def run_single_experiment(config, exp_dir, logger, device):
    logger.info(f"Loading dataset: {config['data']['dataset_name']}")
    train_dataset, test_dataset = get_omniglot_dataset(config)
    
    logger.info("Creating task loaders")
    train_loader = create_task_loader(train_dataset, config, mode="train")
    test_loader = create_task_loader(test_dataset, config, mode="test")
    
    logger.info(f"Creating model: {config['model']['type']}")
    model = create_model(config)
    model = model.to(device)
    logger.info(f"Model parameters: {count_parameters(model)}")
    
    logger.info("Starting training")
    metrics_history = train_model(model, train_loader, config, device, logger)
    
    logger.info("Starting evaluation")
    test_loss, test_acc = evaluate_model(model, test_loader, config, device)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    if config['results'].get('save_checkpoints', True):
        save_checkpoint(model, os.path.join(exp_dir, 'checkpoints'))
    
    final_metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
    }
    
    return metrics_history, final_metrics


def main():
    parser = argparse.ArgumentParser(description="Run a few-shot learning experiment")
    parser.add_argument('--config', required=True, help="Path to configuration file")
    parser.add_argument('--exp_id', default=None, help="Experiment ID (auto-generated if not provided)")
    args = parser.parse_args()
    
    exp_id = args.exp_id if args.exp_id else generate_exp_id()
    config = load_config(args.config)
    config['experiment']['exp_id'] = exp_id
    
    logger = setup_logger(exp_id, config['logging']['log_dir'], config['logging']['level'])
    logger.info(f"Starting experiment: {exp_id}")
    logger.info(f"Experiment name: {config['experiment']['name']}")
    
    device = get_device(config)
    logger.info(f"Using device: {device}")
    
    exp_dir = setup_results_dir(exp_id, config['results']['save_dir'])
    save_config(config, os.path.join(exp_dir, 'config_used.yaml'))
    
    repeat_times = config['experiment']['repeat_times']
    repeat_results = []
    
    try:
        for repeat in range(repeat_times):
            logger.info(f"Starting repeat {repeat + 1}/{repeat_times}")
            set_seed(config['experiment']['seed'] + repeat)
            
            if repeat_times > 1:
                repeat_dir = os.path.join(exp_dir, 'repeats', f'repeat_{repeat + 1:03d}')
                os.makedirs(repeat_dir, exist_ok=True)
            else:
                repeat_dir = exp_dir
            
            metrics_history, final_metrics = run_single_experiment(config, repeat_dir, logger, device)
            repeat_results.append(final_metrics)
            
            if repeat_times == 1:
                save_experiment_results(config, metrics_history, final_metrics, exp_dir)
        
        if repeat_times > 1:
            aggregated = save_repeat_results(repeat_results, exp_dir)
            save_experiment_results(config, [], aggregated, exp_dir)
            logger.info(f"Aggregated results: {aggregated}")
        
        logger.info("Experiment completed successfully")
    
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
