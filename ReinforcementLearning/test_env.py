import sys
print(f"Python version: {sys.version}")

try:
    import gymnasium as gym
    print(f"gymnasium version: {gym.__version__}")
except ImportError as e:
    print(f"gymnasium import error: {e}")

try:
    from stable_baselines3 import DQN
    print("stable-baselines3 DQN imported successfully")
except ImportError as e:
    print(f"stable-baselines3 import error: {e}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"torch import error: {e}")

try:
    env = gym.make('CartPole-v1')
    obs, info = env.reset()
    print(f"CartPole environment reset successful, observation shape: {obs.shape}")
    env.close()
except Exception as e:
    print(f"CartPole env error: {e}")

print("Environment test completed!")