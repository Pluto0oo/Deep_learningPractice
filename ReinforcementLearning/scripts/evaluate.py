"""
实验评估脚本

功能:
- 评估DQN、BC、PER-DQN三种方法
- 生成对比结果和可视化图表
- 保存评估结果供后续分析
- 支持使用缓存结果（当模型加载失败时）
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.evaluator import Evaluator
from src.visualization import ResultVisualizer
from src.config import load_config, get_default_config


def load_cached_results(results_dir):
    """加载已有的评估结果作为缓存"""
    cached_path = os.path.join(results_dir, 'evaluation_results.npy')
    if os.path.exists(cached_path):
        cached = np.load(cached_path, allow_pickle=True).item()
        print(f"Loaded cached evaluation results from: {cached_path}")
        return cached
    return None


def main(config_path=None):
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = get_default_config()

    env_name = config['env']['name']
    eval_config = config['evaluation']
    results_dir = config['results']['save_dir']

    dqn_model_path = os.path.join(results_dir, 'dqn_cartpole.zip')
    bc_model_path = os.path.join(results_dir, 'bc_cartpole.pth')
    per_results_path = os.path.join(results_dir, 'per_results.npy')
    per_history_path = os.path.join(results_dir, 'per_training_history.npy')

    evaluator = Evaluator(env_name=env_name)
    visualizer = ResultVisualizer(save_dir=os.path.join(results_dir, 'plots'))

    # 先加载缓存结果
    cached = load_cached_results(results_dir)
    results = {}

    # 评估DQN（优先使用缓存）
    if cached and 'dqn' in cached:
        results['dqn'] = cached['dqn']
        print(f"DQN (cached): avg={results['dqn']['avg_reward']:.1f} ± {results['dqn']['std_reward']:.1f}")
    elif os.path.exists(dqn_model_path):
        try:
            from stable_baselines3 import DQN
            print(f"Loading DQN model from: {dqn_model_path}")
            dqn_model = DQN.load(dqn_model_path)
            results['dqn'] = evaluator.evaluate_dqn(
                dqn_model,
                num_episodes=eval_config['num_episodes'],
                deterministic=eval_config['deterministic']
            )
            print(f"DQN evaluation complete: avg={results['dqn']['avg_reward']:.1f}")
        except Exception as e:
            print(f"Warning: DQN model load failed: {e}")
            print("Using cached results if available...")
            if cached and 'dqn' in cached:
                results['dqn'] = cached['dqn']
    else:
        print(f"Warning: DQN model not found at {dqn_model_path}")
        if cached and 'dqn' in cached:
            results['dqn'] = cached['dqn']

    # 评估BC
    if cached and 'bc' in cached:
        results['bc'] = cached['bc']
        print(f"BC (cached): avg={results['bc']['avg_reward']:.1f} ± {results['bc']['std_reward']:.1f}")
    elif os.path.exists(bc_model_path):
        try:
            from src.models import BehavioralCloningNet
            print(f"\nLoading BC model from: {bc_model_path}")
            expert_data_path = config['expert_data']['save_path']
            expert_data = np.load(expert_data_path, allow_pickle=True)
            state_dim = expert_data[0]['state'].shape[0]

            bc_model = BehavioralCloningNet(state_dim=state_dim, action_dim=2)
            bc_model.load_state_dict(torch.load(bc_model_path, map_location='cpu'))

            results['bc'] = evaluator.evaluate_bc(
                bc_model,
                num_episodes=eval_config['num_episodes']
            )
            print(f"BC evaluation complete: avg={results['bc']['avg_reward']:.1f}")
        except Exception as e:
            print(f"Warning: BC model load failed: {e}")
            if cached and 'bc' in cached:
                results['bc'] = cached['bc']
    else:
        print(f"Warning: BC model not found at {bc_model_path}")
        if cached and 'bc' in cached:
            results['bc'] = cached['bc']

    # 加载PER-DQN结果
    if os.path.exists(per_results_path):
        per_results = np.load(per_results_path, allow_pickle=True).item()
        results['per_dqn'] = per_results
        print(f"PER-DQN: avg={results['per_dqn']['avg_reward']:.1f} ± {results['per_dqn']['std_reward']:.1f}")
    else:
        print("Warning: PER results not found. Run train_prioritized_dqn.py first.")

    # 生成可视化和保存结果
    if results:
        print("\n" + "=" * 60)
        print("Evaluation Results Summary")
        print("=" * 60)
        evaluator.print_comparison(results)

        # 加载训练历史
        bc_train_history = None
        bc_loss_path = os.path.join(results_dir, 'bc_loss_history.npy')
        bc_acc_path = os.path.join(results_dir, 'bc_accuracy_history.npy')
        if os.path.exists(bc_loss_path) and os.path.exists(bc_acc_path):
            bc_loss = np.load(bc_loss_path).tolist()
            bc_acc = np.load(bc_acc_path).tolist()
            bc_train_history = (bc_loss, bc_acc)

        per_train_history = None
        if os.path.exists(per_history_path):
            per_train_history = np.load(per_history_path, allow_pickle=True).item()

        # 生成所有图表
        visualizer.generate_all_plots(
            results,
            bc_train_history=bc_train_history,
            per_train_history=per_train_history,
        )

        # 保存合并结果
        results_save_path = os.path.join(results_dir, 'evaluation_results.npy')
        np.save(results_save_path, results)
        print(f"\nResults saved to {results_save_path}")

        # 打印对比总结
        print("\n" + "=" * 60)
        print("Method Comparison Summary")
        print("=" * 60)
        for method, metrics in results.items():
            print(f"\n{method.upper()}:")
            print(f"  Average Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
            print(f"  Max Reward:     {metrics['max_reward']:.2f}")
            print(f"  Min Reward:     {metrics['min_reward']:.2f}")
            print(f"  Average Length: {metrics['avg_length']:.2f} ± {metrics['std_length']:.2f}")

        # DQN vs PER-DQN对比
        if 'dqn' in results and 'per_dqn' in results:
            dqn_avg = results['dqn']['avg_reward']
            per_avg = results['per_dqn']['avg_reward']
            diff = per_avg - dqn_avg
            pct = (diff / dqn_avg * 100) if dqn_avg != 0 else 0
            print(f"\nPER-DQN vs DQN Comparison:")
            print(f"  DQN Average:     {dqn_avg:.2f}")
            print(f"  PER-DQN Average: {per_avg:.2f}")
            print(f"  Difference:      {diff:+.2f} ({pct:+.1f}%)")
        print("=" * 60)
    else:
        print("\nNo results available for evaluation")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate DQN, BC, and PER-DQN models")
    parser.add_argument('--config', type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    main(args.config)
