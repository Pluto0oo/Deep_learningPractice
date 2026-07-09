import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def save_results(results_history: List[Dict], exp_dir: str) -> None:
    os.makedirs(exp_dir, exist_ok=True)

    results_path = os.path.join(exp_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results_history, f, indent=4)

    print(f"Results saved to {results_path}")


def generate_plot(results_history: List[Dict], exp_dir: str, title: str = "FedAvg Training Curve") -> None:
    os.makedirs(exp_dir, exist_ok=True)

    rounds = [r['round'] for r in results_history]
    test_loss = [r['test_loss'] for r in results_history]
    test_acc = [r['test_acc'] for r in results_history]

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rounds, test_loss, marker='o', color='#1f77b4', label='Test Loss')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Loss')
    plt.title('Test Loss vs Communication Rounds')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rounds, test_acc, marker='o', color='#ff7f0e', label='Test Accuracy')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy vs Communication Rounds')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(exp_dir, 'training_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {plot_path}")


def generate_summary(config: Dict, results_history: List[Dict], exp_dir: str) -> None:
    os.makedirs(exp_dir, exist_ok=True)

    summary = {
        'config': config,
        'final_results': results_history[-1] if results_history else {},
        'best_results': max(results_history, key=lambda x: x['test_acc']) if results_history else {},
        'avg_acc': np.mean([r['test_acc'] for r in results_history]) if results_history else 0,
        'std_acc': np.std([r['test_acc'] for r in results_history]) if results_history else 0,
    }

    summary_path = os.path.join(exp_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"Summary saved to {summary_path}")


def save_experiment_results(config: Dict, results_history: List[Dict], exp_dir: str) -> None:
    save_results(results_history, exp_dir)
    generate_summary(config, results_history, exp_dir)

    if len(results_history) > 0:
        title = f"{config['experiment'].get('algorithm', 'fedavg').upper()} Training Curve"
        generate_plot(results_history, exp_dir, title)