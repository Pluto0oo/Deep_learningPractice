import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.models import BehavioralCloningNet
from src.trainer import BehavioralCloningTrainer
from src.config import load_config, get_default_config


def main(config_path=None):
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = get_default_config()

    env_name = config['env']['name']
    bc_config = config['bc']
    expert_data_path = config['expert_data']['save_path']
    results_dir = config['results']['save_dir']

    if not os.path.exists(expert_data_path):
        print(f"Error: Expert data not found at {expert_data_path}")
        print("Please collect expert data first using collect_expert_data.py")
        return

    print(f"Loading expert data from: {expert_data_path}")
    expert_data = np.load(expert_data_path, allow_pickle=True)

    states = np.array([step['state'] for step in expert_data])
    actions = np.array([step['action'] for step in expert_data])

    print(f"Loaded {len(states)} samples")
    print(f"State shape: {states.shape}")
    print(f"Action shape: {actions.shape}")

    state_dim = states.shape[1]
    action_dim = len(np.unique(actions))

    print(f"\nState dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    model = BehavioralCloningNet(state_dim=state_dim, action_dim=action_dim, hidden_dim=bc_config['hidden_dim'])
    trainer = BehavioralCloningTrainer(model, device='cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nTraining Behavioral Cloning model...")
    print(f"Epochs: {bc_config['epochs']}")
    print(f"Batch size: {bc_config['batch_size']}")
    print(f"Learning rate: {bc_config['learning_rate']}")

    loss_history, accuracy_history = trainer.train(
        states=states,
        actions=actions,
        epochs=bc_config['epochs'],
        batch_size=bc_config['batch_size'],
        lr=bc_config['learning_rate']
    )

    model_path = os.path.join(results_dir, 'bc_cartpole.pth')
    trainer.save_model(model_path)
    print(f"\nBC model saved to {model_path}")

    np.save(os.path.join(results_dir, 'bc_loss_history.npy'), loss_history)
    np.save(os.path.join(results_dir, 'bc_accuracy_history.npy'), accuracy_history)
    print("Training history saved")

    print("\nBehavioral Cloning training complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Behavioral Cloning model")
    parser.add_argument('--config', type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    main(args.config)
