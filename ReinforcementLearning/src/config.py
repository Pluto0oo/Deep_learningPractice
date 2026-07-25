import yaml
import os


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_default_config():
    return {
        'env': {
            'name': 'CartPole-v1',
            'max_episode_steps': 500
        },
        'dqn': {
            'policy': 'MlpPolicy',
            'learning_rate': 0.001,
            'buffer_size': 1000000,
            'learning_starts': 50000,
            'batch_size': 64,
            'gamma': 0.99,
            'target_update_interval': 10000,
            'train_freq': 4,
            'gradient_steps': 1,
            'total_timesteps': 1000000,
            'verbose': 1
        },
        'bc': {
            'hidden_dim': 128,
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        'expert_data': {
            'num_episodes': 100,
            'save_path': './data/expert_data.npy'
        },
        'evaluation': {
            'num_episodes': 10,
            'deterministic': True
        },
        'results': {
            'save_dir': './results',
            'save_models': True
        }
    }


def save_config(config, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
