import os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class Visualizer:
    def __init__(self, config):
        self.config = config
        self.plots_dir = config.get('visualization.plots_dir', './results/plots')
        self.figsize = tuple(config.get('visualization.figsize', [10, 6]))
        self.dpi = config.get('visualization.dpi', 300)
        
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_training_curves(self, histories, method_names, filename='training_curves.png'):
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        for history, method in zip(histories, method_names):
            train_loss = [h['loss'] for h in history['train']]
            val_loss = [h['loss'] for h in history['val']]
            train_acc = [h['accuracy'] for h in history['train']]
            val_acc = [h['accuracy'] for h in history['val']]
            
            epochs = range(1, len(train_loss) + 1)
            
            plt.subplot(1, 2, 1)
            plt.plot(epochs, train_loss, label=f'{method} Train')
            plt.plot(epochs, val_loss, label=f'{method} Val')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(epochs, train_acc, label=f'{method} Train')
            plt.plot(epochs, val_acc, label=f'{method} Val')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.legend()
        
        plt.tight_layout()
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Training curves saved to {filepath}")

    def plot_task_accuracies(self, all_task_accuracies, method_names, task_names, filename='task_accuracies.png'):
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        for method_idx, (task_accuracies, method) in enumerate(zip(all_task_accuracies, method_names)):
            x = range(1, len(task_names) + 1)
            
            for task_name in task_names:
                accuracies = []
                for step in task_accuracies:
                    if task_name in step:
                        accuracies.append(step[task_name])
                    else:
                        accuracies.append(None)
                
                plt.plot(x, accuracies, marker='o', label=f'{method} - {task_name}')
        
        plt.xlabel('Task')
        plt.ylabel('Accuracy')
        plt.title('Task Accuracies Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Task accuracies plot saved to {filepath}")

    def plot_forgetting_rates(self, forgetting_rates, method_names, filename='forgetting_rates.png'):
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        bar_width = 0.2
        tasks = list(forgetting_rates[0].keys())
        x = np.arange(len(tasks))
        
        for i, (rates, method) in enumerate(zip(forgetting_rates, method_names)):
            values = [rates[task] for task in tasks]
            plt.bar(x + i * bar_width, values, width=bar_width, label=method)
        
        plt.xlabel('Task')
        plt.ylabel('Forgetting Rate')
        plt.title('Forgetting Rates by Method')
        plt.xticks(x + bar_width * (len(method_names) - 1) / 2, tasks)
        plt.legend()
        plt.tight_layout()
        
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Forgetting rates plot saved to {filepath}")

    def plot_comparison_bar(self, metrics, method_names, metric_name, filename='comparison.png'):
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        x = np.arange(len(method_names))
        values = [m[metric_name] for m in metrics]
        
        plt.bar(x, values)
        plt.xlabel('Method')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Comparison')
        plt.xticks(x, method_names)
        
        for i, v in enumerate(values):
            plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Comparison plot saved to {filepath}")

    def plot_memory_usage(self, memory_usages, method_names, filename='memory_usage.png'):
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        x = np.arange(len(method_names))
        bar_width = 0.4
        
        allocated = [m['allocated_mb'] for m in memory_usages]
        reserved = [m['reserved_mb'] for m in memory_usages]
        
        plt.bar(x - bar_width/2, allocated, width=bar_width, label='Allocated')
        plt.bar(x + bar_width/2, reserved, width=bar_width, label='Reserved')
        
        plt.xlabel('Method')
        plt.ylabel('Memory (MB)')
        plt.title('Memory Usage Comparison')
        plt.xticks(x, method_names)
        plt.legend()
        plt.tight_layout()
        
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Memory usage plot saved to {filepath}")

    def plot_runtime_comparison(self, runtimes, method_names, filename='runtime_comparison.png'):
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        x = np.arange(len(method_names))
        values = [r for r in runtimes]
        
        plt.bar(x, values)
        plt.xlabel('Method')
        plt.ylabel('Runtime (seconds)')
        plt.title('Training Runtime Comparison')
        plt.xticks(x, method_names)
        
        for i, v in enumerate(values):
            plt.text(i, v, f'{v:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Runtime comparison plot saved to {filepath}")