import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from src.config import load_config, get_default_config


def main(config_path=None):
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = get_default_config()

    env_name = config['env']['name']
    results_dir = config['results']['save_dir']
    
    os.makedirs(results_dir, exist_ok=True)

    print("Training Prioritized DQN on CartPole-v1...")
    print("Paper Reference: \"Prioritized Experience Replay\" (Schaul et al., 2015)")

    model = DQN(
        policy='MlpPolicy',
        env=env_name,
        learning_rate=0.001,
        buffer_size=1000000,
        learning_starts=50000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=10000,
        train_freq=4,
        gradient_steps=1,
        verbose=1,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6
    )

    model.learn(total_timesteps=1000000)

    model_path = os.path.join(results_dir, 'prioritized_dqn_cartpole.zip')
    model.save(model_path)
    print(f"Prioritized DQN model saved to {model_path}")

    print("\nPrioritized DQN training complete!")
    print("Key improvements over standard DQN:")
    print("- Experience replay with prioritized sampling")
    print("- TD-error based prioritization (alpha=0.6)")
    print("- Importance sampling weights (beta=0.4)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Prioritized DQN")
    parser.add_argument('--config', type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    main(args.config)
