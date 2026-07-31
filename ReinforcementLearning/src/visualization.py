"""
实验结果可视化模块

功能:
- 多方法奖励对比图（柱状图+误差棒+数据标注）
- 奖励分布箱线图（含散点叠加）
- 95%置信区间图（含重叠区域标注）
- 训练曲线图（含平滑曲线和置信带）
- 综合对比面板（多子图布局）
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# 全局样式设置
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# 专业配色方案
COLORS = {
    'dqn': '#2196F3',        # 蓝色
    'per_dqn': '#FF9800',    # 橙色
    'bc': '#4CAF50',          # 绿色
    'random': '#9E9E9E',     # 灰色
}

METHOD_LABELS = {
    'dqn': 'DQN',
    'per_dqn': 'PER-DQN',
    'bc': 'BC',
    'random': 'Random',
}


class ResultVisualizer:
    """实验结果可视化器"""

    def __init__(self, save_dir='./results/plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _get_color(self, method):
        return COLORS.get(method, '#795548')

    def _get_label(self, method):
        return METHOD_LABELS.get(method, method.upper())

    def plot_reward_comparison(self, results, title='Reward Comparison', filename='reward_comparison.png'):
        """改进的奖励对比图：柱状图+误差棒+数据标注"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        methods = list(results.keys())
        labels = [self._get_label(m) for m in methods]
        colors = [self._get_color(m) for m in methods]

        # 左图：平均奖励
        avg_rewards = [results[m]['avg_reward'] for m in methods]
        std_rewards = [results[m]['std_reward'] for m in methods]

        bars = axes[0].bar(labels, avg_rewards, yerr=std_rewards, capsize=8,
                          color=colors, edgecolor='black', linewidth=0.8,
                          error_kw={'linewidth': 2, 'capthick': 2})

        # 添加数据标注
        for bar, avg, std in zip(bars, avg_rewards, std_rewards):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2., height + std + 3,
                        f'{avg:.1f}±{std:.1f}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

        axes[0].set_title('Average Reward (10 episodes)', fontweight='bold')
        axes[0].set_ylabel('Reward')
        axes[0].set_ylim(0, max([a + s for a, s in zip(avg_rewards, std_rewards)]) * 1.2)

        # 添加基准线（CartPole最大值500）
        axes[0].axhline(y=500, color='red', linestyle='--', alpha=0.5, label='Max (500)')
        axes[0].legend(loc='upper left', fontsize=9)

        # 右图：平均回合长度
        avg_lengths = [results[m]['avg_length'] for m in methods]
        std_lengths = [results[m]['std_length'] for m in methods]

        bars2 = axes[1].bar(labels, avg_lengths, yerr=std_lengths, capsize=8,
                           color=colors, edgecolor='black', linewidth=0.8,
                           error_kw={'linewidth': 2, 'capthick': 2})

        for bar, avg, std in zip(bars2, avg_lengths, std_lengths):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2., height + std + 3,
                        f'{avg:.1f}±{std:.1f}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

        axes[1].set_title('Average Episode Length', fontweight='bold')
        axes[1].set_ylabel('Steps')
        axes[1].set_ylim(0, max([a + s for a, s in zip(avg_lengths, std_lengths)]) * 1.2)

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
        print(f"  Saved: {filename}")

    def plot_reward_distribution(self, results, filename='reward_distribution.png'):
        """改进的奖励分布图：箱线图+散点叠加"""
        n = len(results)
        fig, ax = plt.subplots(figsize=(10, 6))

        data = []
        labels = []
        colors = []
        for method, metrics in results.items():
            data.append(metrics['rewards'])
            labels.append(self._get_label(method))
            colors.append(self._get_color(method))

        # 箱线图
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                       showmeans=True, meanprops={'marker': 'D', 'markerfacecolor': 'white',
                                                   'markeredgecolor': 'black', 'markersize': 7})

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # 叠加散点
        for i, (method, metrics) in enumerate(results.items()):
            x = np.random.normal(i + 1, 0.04, size=len(metrics['rewards']))
            ax.scatter(x, metrics['rewards'], alpha=0.7, color=colors[i],
                      edgecolor='black', linewidth=0.5, s=50, zorder=3)

        ax.set_title('Reward Distribution by Method', fontweight='bold')
        ax.set_ylabel('Reward')
        ax.set_xlabel('Method')

        # 添加最大值基准线
        ax.axhline(y=500, color='red', linestyle='--', alpha=0.5, label='Max Possible (500)')
        ax.legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
        print(f"  Saved: {filename}")

    def plot_confidence_interval(self, results, filename='confidence_interval.png'):
        """改进的置信区间图：含重叠区域标注"""
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = list(results.keys())
        labels = [self._get_label(m) for m in methods]
        colors = [self._get_color(m) for m in methods]

        avg_rewards = [results[m]['avg_reward'] for m in methods]
        std_rewards = [results[m]['std_reward'] for m in methods]
        n = len(list(results.values())[0]['rewards'])

        # 95% CI
        ci_lower = [avg - 1.96 * (std / np.sqrt(n)) for avg, std in zip(avg_rewards, std_rewards)]
        ci_upper = [avg + 1.96 * (std / np.sqrt(n)) for avg, std in zip(avg_rewards, std_rewards)]

        x = np.arange(len(methods))

        # 误差棒
        for i in range(len(methods)):
            ax.errorbar(x[i], avg_rewards[i],
                       yerr=[[avg_rewards[i] - ci_lower[i]], [ci_upper[i] - avg_rewards[i]]],
                       fmt='o', color=colors[i], markersize=12, capsize=8,
                       capthick=2, elinewidth=2, label=labels[i])

            # 标注数值
            ax.text(x[i] + 0.1, avg_rewards[i],
                   f'{avg_rewards[i]:.1f}\n[{ci_lower[i]:.1f}, {ci_upper[i]:.1f}]',
                   fontsize=9, va='center')

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title('Reward with 95% Confidence Interval', fontweight='bold')
        ax.set_ylabel('Average Reward')
        ax.set_xlabel('Method')
        ax.axhline(y=500, color='red', linestyle='--', alpha=0.5, label='Max (500)')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(-0.5, len(methods) - 0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
        print(f"  Saved: {filename}")

    def plot_training_curve(self, loss_history, accuracy_history, title='BC Training Curve', filename='bc_training.png'):
        """改进的BC训练曲线：含平滑和填充"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(loss_history) + 1)

        # 损失曲线
        axes[0].plot(epochs, loss_history, color=COLORS['bc'], alpha=0.3, linewidth=1)
        # 平滑曲线
        if len(loss_history) > 10:
            window = min(10, len(loss_history) // 5)
            smoothed = np.convolve(loss_history, np.ones(window) / window, mode='valid')
            axes[0].plot(range(window, window + len(smoothed)), smoothed,
                        color=COLORS['bc'], linewidth=2, label='Smoothed')
        else:
            axes[0].plot(epochs, loss_history, color=COLORS['bc'], linewidth=2)

        axes[0].set_title(f'{title} - Loss', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        # 准确率曲线
        axes[1].plot(epochs, accuracy_history, color=COLORS['dqn'], alpha=0.3, linewidth=1)
        if len(accuracy_history) > 10:
            window = min(10, len(accuracy_history) // 5)
            smoothed = np.convolve(accuracy_history, np.ones(window) / window, mode='valid')
            axes[1].plot(range(window, window + len(smoothed)), smoothed,
                        color=COLORS['dqn'], linewidth=2, label='Smoothed')
        else:
            axes[1].plot(epochs, accuracy_history, color=COLORS['dqn'], linewidth=2)

        # 标注最终值
        final_acc = accuracy_history[-1]
        axes[1].axhline(y=final_acc, color='gray', linestyle=':', alpha=0.5)
        axes[1].text(len(accuracy_history) * 0.7, final_acc + 0.01,
                    f'Final: {final_acc:.4f}', fontsize=10, color=COLORS['dqn'])

        axes[1].set_title(f'{title} - Accuracy', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
        print(f"  Saved: {filename}")

    def plot_per_training_curve(self, training_history, filename='per_training_curve.png'):
        """PER-DQN训练曲线：奖励变化+损失变化"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 奖励曲线（滑动平均）
        episode_rewards = training_history['episode_rewards']
        episodes = range(1, len(episode_rewards) + 1)

        axes[0].plot(episodes, episode_rewards, color=COLORS['per_dqn'], alpha=0.2, linewidth=0.5)

        # 滑动平均
        window = min(50, max(1, len(episode_rewards) // 10))
        if len(episode_rewards) > window:
            smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
            axes[0].plot(range(window, window + len(smoothed)), smoothed,
                        color=COLORS['per_dqn'], linewidth=2, label=f'MA-{window}')

        # 评估点
        eval_history = training_history.get('eval_rewards_history', [])
        if eval_history:
            eval_steps = [e['timestep'] for e in eval_history]
            eval_rewards = [e['avg_reward'] for e in eval_history]
            eval_stds = [e['std_reward'] for e in eval_history]
            # 归一化评估点到回合数
            ax2 = axes[0].twinx()
            ax2.plot(eval_steps, eval_rewards, 'D-', color='red', markersize=5, label='Eval')
            ax2.fill_between(eval_steps,
                            [r - s for r, s in zip(eval_rewards, eval_stds)],
                            [r + s for r, s in zip(eval_rewards, eval_stds)],
                            color='red', alpha=0.1)
            ax2.set_ylabel('Evaluation Reward', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.legend(loc='upper left', fontsize=8)

        axes[0].set_title('PER-DQN Training - Episode Rewards', fontweight='bold')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Episode Reward')
        axes[0].legend(loc='lower right', fontsize=9)

        # 损失曲线
        losses = training_history['losses']
        if losses:
            steps = range(1, len(losses) + 1)
            axes[1].plot(steps, losses, color=COLORS['dqn'], alpha=0.2, linewidth=0.5)

            # 平滑
            window = min(100, max(1, len(losses) // 10))
            if len(losses) > window:
                smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
                axes[1].plot(range(window, window + len(smoothed)), smoothed,
                            color=COLORS['dqn'], linewidth=2, label=f'MA-{window}')

            axes[1].set_title('PER-DQN Training - Loss', fontweight='bold')
            axes[1].set_xlabel('Training Step')
            axes[1].set_ylabel('Huber Loss')
            axes[1].legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
        print(f"  Saved: {filename}")

    def plot_comprehensive_comparison(self, results, filename='comprehensive_comparison.png'):
        """综合对比面板：4个子图展示不同维度"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        methods = list(results.keys())
        labels = [self._get_label(m) for m in methods]
        colors = [self._get_color(m) for m in methods]

        # 1. 平均奖励对比
        avg_rewards = [results[m]['avg_reward'] for m in methods]
        std_rewards = [results[m]['std_reward'] for m in methods]

        bars = axes[0, 0].bar(labels, avg_rewards, yerr=std_rewards, capsize=6,
                              color=colors, edgecolor='black', linewidth=0.8)
        for bar, avg in zip(bars, avg_rewards):
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 2,
                          f'{avg:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        axes[0, 0].set_title('Average Reward', fontweight='bold')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].axhline(y=500, color='red', linestyle='--', alpha=0.3)

        # 2. 奖励分布箱线图
        data = [results[m]['rewards'] for m in methods]
        bp = axes[0, 1].boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[0, 1].set_title('Reward Distribution', fontweight='bold')
        axes[0, 1].set_ylabel('Reward')

        # 3. 最大/最小/平均奖励对比
        max_rewards = [results[m]['max_reward'] for m in methods]
        min_rewards = [results[m]['min_reward'] for m in methods]

        x = np.arange(len(methods))
        width = 0.25

        axes[1, 0].bar(x - width, min_rewards, width, label='Min', color='#EF5350', edgecolor='black')
        axes[1, 0].bar(x, avg_rewards, width, label='Avg', color='#66BB6A', edgecolor='black')
        axes[1, 0].bar(x + width, max_rewards, width, label='Max', color='#42A5F5', edgecolor='black')

        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(labels)
        axes[1, 0].set_title('Min/Avg/Max Reward', fontweight='bold')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].legend()

        # 4. 稳定性对比（标准差+变异系数）
        std_rewards = [results[m]['std_reward'] for m in methods]
        cv = [results[m]['std_reward'] / results[m]['avg_reward'] * 100 for m in methods]

        ax4 = axes[1, 1]
        bars4 = ax4.bar(labels, cv, color=colors, edgecolor='black', linewidth=0.8)
        for bar, val in zip(bars4, cv):
            ax4.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax4.set_title('Coefficient of Variation (Lower = More Stable)', fontweight='bold')
        ax4.set_ylabel('CV (%)')

        plt.suptitle('Comprehensive Method Comparison', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
        print(f"  Saved: {filename}")

    def generate_all_plots(self, results, bc_train_history=None, per_train_history=None):
        """生成所有可视化图表"""
        print(f"\nGenerating visualization plots in {self.save_dir}...")

        # 基础对比图
        self.plot_reward_comparison(results, filename='reward_comparison.png')
        self.plot_reward_distribution(results, filename='reward_distribution.png')
        self.plot_confidence_interval(results, filename='confidence_interval.png')

        # 综合对比面板
        self.plot_comprehensive_comparison(results, filename='comprehensive_comparison.png')

        # BC训练曲线
        if bc_train_history:
            loss_hist, acc_hist = bc_train_history
            self.plot_training_curve(loss_hist, acc_hist, filename='bc_training.png')

        # PER训练曲线
        if per_train_history:
            self.plot_per_training_curve(per_train_history, filename='per_training_curve.png')

        print(f"All plots saved to {self.save_dir}")
