import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class ResultVisualizer:
    def __init__(self, save_dir='./results/plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        sns.set_style('whitegrid')
        plt.rcParams['font.size'] = 12
        plt.rcParams['figure.dpi'] = 100

    def plot_reward_comparison(self, results, title='Reward Comparison', filename='reward_comparison.png'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        methods = list(results.keys())
        avg_rewards = [results[m]['avg_reward'] for m in methods]
        std_rewards = [results[m]['std_reward'] for m in methods]

        axes[0].bar(methods, avg_rewards, yerr=std_rewards, capsize=5)
        axes[0].set_title('Average Reward')
        axes[0].set_ylabel('Reward')

        avg_lengths = [results[m]['avg_length'] for m in methods]
        std_lengths = [results[m]['std_length'] for m in methods]

        axes[1].bar(methods, avg_lengths, yerr=std_lengths, capsize=5)
        axes[1].set_title('Average Episode Length')
        axes[1].set_ylabel('Length')

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

    def plot_training_curve(self, loss_history, accuracy_history, title='BC Training Curve', filename='bc_training.png'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(loss_history)
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')

        axes[1].plot(accuracy_history)
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

    def plot_reward_distribution(self, results, filename='reward_distribution.png'):
        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))

        if len(results) == 1:
            axes = [axes]

        for i, (method, metrics) in enumerate(results.items()):
            axes[i].hist(metrics['rewards'], bins=10)
            axes[i].set_title(f'{method.upper()} Reward Distribution')
            axes[i].set_xlabel('Reward')
            axes[i].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

    def plot_confidence_interval(self, results, filename='confidence_interval.png'):
        fig, ax = plt.subplots(figsize=(8, 5))

        methods = list(results.keys())
        avg_rewards = [results[m]['avg_reward'] for m in methods]
        std_rewards = [results[m]['std_reward'] for m in methods]
        n = len(list(results.values())[0]['rewards'])
        
        ci_lower = [avg - 1.96 * (std / np.sqrt(n)) for avg, std in zip(avg_rewards, std_rewards)]
        ci_upper = [avg + 1.96 * (std / np.sqrt(n)) for avg, std in zip(avg_rewards, std_rewards)]

        x = np.arange(len(methods))
        ax.errorbar(x, avg_rewards, yerr=[np.array(avg_rewards) - np.array(ci_lower), 
                                          np.array(ci_upper) - np.array(avg_rewards)],
                    fmt='o-', capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title('Reward with 95% Confidence Interval')
        ax.set_ylabel('Reward')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

    def generate_all_plots(self, results, bc_train_history=None):
        self.plot_reward_comparison(results, filename='reward_comparison.png')
        self.plot_reward_distribution(results, filename='reward_distribution.png')
        self.plot_confidence_interval(results, filename='confidence_interval.png')

        if bc_train_history:
            loss_hist, acc_hist = bc_train_history
            self.plot_training_curve(loss_hist, acc_hist, filename='bc_training.png')

        print(f"All plots saved to {self.save_dir}")
