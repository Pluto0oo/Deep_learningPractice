import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from src.config import load_config, get_default_config


def setup_logger(log_dir, level=logging.INFO):
    logger = logging.getLogger('dqn_trainer')
    logger.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def main(config_path=None):
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = get_default_config()

    env_name = config['env']['name']
    dqn_params = config['dqn']
    results_dir = config['results']['save_dir']
    
    os.makedirs(results_dir, exist_ok=True)
    
    logger = setup_logger(results_dir)

    logger.info(f"Training DQN on {env_name}...")
    logger.info(f"Total timesteps: {dqn_params['total_timesteps']}")
    logger.info(f"Policy: {dqn_params['policy']}")
    logger.info(f"Learning rate: {dqn_params['learning_rate']}")
    logger.info(f"Batch size: {dqn_params['batch_size']}")

    model = DQN(
        policy=dqn_params['policy'],
        env=env_name,
        learning_rate=dqn_params['learning_rate'],
        buffer_size=dqn_params['buffer_size'],
        learning_starts=dqn_params['learning_starts'],
        batch_size=dqn_params['batch_size'],
        gamma=dqn_params['gamma'],
        target_update_interval=dqn_params['target_update_interval'],
        train_freq=dqn_params['train_freq'],
        gradient_steps=dqn_params['gradient_steps'],
        verbose=dqn_params['verbose']
    )

    logger.info("Starting model training...")
    model.learn(total_timesteps=dqn_params['total_timesteps'])

    model_path = os.path.join(results_dir, 'dqn_cartpole.zip')
    model.save(model_path)
    logger.info(f"DQN model saved to {model_path}")

    logger.info("DQN training complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train DQN on CartPole")
    parser.add_argument('--config', type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    main(args.config)
