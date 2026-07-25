import os
import sys
import logging
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.visualization import Visualizer

logger = logging.getLogger(__name__)


def main(config_path):
    config = Config(config_path)
    visualizer = Visualizer(config)
    
    results_dir = config.get('evaluation.results_dir', './results')
    
    methods = ['fine_tuning', 'ewc', 'lwf', 'der']
    method_names = ['Fine Tuning', 'EWC', 'LWF', 'DER']
    
    all_results = []
    all_task_accuracies = []
    all_forgetting_rates = []
    all_runtimes = []
    all_memory_usages = []
    
    for method in methods:
        filepath = os.path.join(results_dir, f'{method}_results.npy')
        if os.path.exists(filepath):
            results = np.load(filepath, allow_pickle=True).item()
            all_results.append(results)
            all_task_accuracies.append(results['task_accuracies'])
            all_forgetting_rates.append(results['forgetting_rates'])
            all_runtimes.append(results['total_time'])
            all_memory_usages.append(results['memory_usage'])
            logger.info(f"Loaded results for {method}: {results['final_accuracies']}")
        else:
            logger.warning(f"Results file not found: {filepath}")
    
    if len(all_results) == 0:
        logger.error("No results found. Please run the individual methods first.")
        return
    
    logger.info(f"\n{'=' * 60}")
    logger.info("COMPARISON RESULTS")
    logger.info(f"{'=' * 60}")
    
    task_names = list(all_forgetting_rates[0].keys())
    
    logger.info("\n1. Final Accuracies:")
    logger.info("-" * 40)
    for method, results in zip(method_names, all_results):
        logger.info(f"{method}:")
        for task, acc in results['final_accuracies'].items():
            logger.info(f"  {task}: {acc:.4f}")
    
    logger.info("\n2. Forgetting Rates:")
    logger.info("-" * 40)
    for method, rates in zip(method_names, all_forgetting_rates):
        logger.info(f"{method}:")
        for task, rate in rates.items():
            logger.info(f"  {task}: {rate:.4f}")
    
    logger.info("\n3. Training Time:")
    logger.info("-" * 40)
    for method, runtime in zip(method_names, all_runtimes):
        logger.info(f"{method}: {runtime:.2f}s")
    
    logger.info("\n4. Memory Usage:")
    logger.info("-" * 40)
    for method, memory in zip(method_names, all_memory_usages):
        logger.info(f"{method}:")
        logger.info(f"  Allocated: {memory['allocated_mb']:.2f}MB")
        logger.info(f"  Reserved: {memory['reserved_mb']:.2f}MB")
    
    visualizer.plot_task_accuracies(
        all_task_accuracies,
        method_names,
        task_names,
        filename='method_comparison.png'
    )
    
    visualizer.plot_forgetting_rates(
        all_forgetting_rates,
        method_names,
        filename='forgetting_comparison.png'
    )
    
    visualizer.plot_runtime_comparison(
        all_runtimes,
        method_names,
        filename='runtime_comparison.png'
    )
    
    visualizer.plot_memory_usage(
        all_memory_usages,
        method_names,
        filename='memory_comparison.png'
    )
    
    avg_accuracies = []
    for results in all_results:
        avg_acc = sum(results['final_accuracies'].values()) / len(results['final_accuracies'])
        avg_accuracies.append({'accuracy': avg_acc})
    
    visualizer.plot_comparison_bar(
        avg_accuracies,
        method_names,
        'accuracy',
        filename='accuracy_comparison.png'
    )
    
    logger.info("\nVisualizations saved to ./results/plots/")
    
    comparison_summary = {
        'methods': methods,
        'method_names': method_names,
        'results': all_results,
        'task_names': task_names
    }
    
    np.save(os.path.join(results_dir, 'comparison_summary.npy'), comparison_summary)
    logger.info("Comparison summary saved to ./results/comparison_summary.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Comparison of All Methods')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)