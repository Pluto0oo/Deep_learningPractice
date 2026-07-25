import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collector import ExpertDataCollector
from src.config import load_config, get_default_config


def main(config_path=None):
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = get_default_config()

    env_name = config['env']['name']
    expert_data_config = config['expert_data']
    results_dir = config['results']['save_dir']

    model_path = os.path.join(results_dir, 'dqn_cartpole.zip')
    
    print(f"Collecting expert data using model: {model_path}")
    print(f"Environment: {env_name}")
    print(f"Number of episodes: {expert_data_config['num_episodes']}")

    collector = ExpertDataCollector(env_name=env_name, model_path=model_path)
    
    if not collector.load_expert_model():
        print(f"Error: Expert model not found at {model_path}")
        print("Please train DQN first using train_dqn.py")
        return

    expert_data = collector.collect_data(
        num_episodes=expert_data_config['num_episodes'],
        save_path=expert_data_config['save_path']
    )

    stats = collector.get_statistics()
    print("\nExpert Data Statistics:")
    print(f"  Number of episodes: {stats['num_episodes']}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Average reward: {stats['avg_reward']:.2f}")
    print(f"  Reward std: {stats['std_reward']:.2f}")
    print(f"  Max reward: {stats['max_reward']:.2f}")
    print(f"  Min reward: {stats['min_reward']:.2f}")

    collector.close()
    print("\nExpert data collection complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect expert data from DQN")
    parser.add_argument('--config', type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    main(args.config)
