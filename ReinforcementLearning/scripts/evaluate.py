import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from stable_baselines3 import DQN

from src.models import BehavioralCloningNet
from src.evaluator import Evaluator
from src.visualization import ResultVisualizer
from src.config import load_config, get_default_config


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

    evaluator = Evaluator(env_name=env_name)
    visualizer = ResultVisualizer(save_dir=os.path.join(results_dir, 'plots'))

    results = {}

    if os.path.exists(dqn_model_path):
        print(f"Loading DQN model from: {dqn_model_path}")
        dqn_model = DQN.load(dqn_model_path)
        results['dqn'] = evaluator.evaluate_dqn(
            dqn_model, 
            num_episodes=eval_config['num_episodes'],
            deterministic=eval_config['deterministic']
        )
        print(f"DQN evaluation complete")
    else:
        print(f"Warning: DQN model not found at {dqn_model_path}")

    if os.path.exists(bc_model_path):
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
        print(f"BC evaluation complete")
    else:
        print(f"Warning: BC model not found at {bc_model_path}")

    if results:
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        evaluator.print_comparison(results)

        visualizer.generate_all_plots(results)

        results_save_path = os.path.join(results_dir, 'evaluation_results.npy')
        np.save(results_save_path, results)
        print(f"\nResults saved to {results_save_path}")
    else:
        print("\nNo models found for evaluation")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate DQN and BC models")
    parser.add_argument('--config', type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    main(args.config)
