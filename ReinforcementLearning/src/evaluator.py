import numpy as np
import gymnasium as gym
import torch


class Evaluator:
    def __init__(self, env_name='CartPole-v1'):
        self.env_name = env_name

    def evaluate_dqn(self, model, num_episodes=10, deterministic=True, render=False):
        env = gym.make(self.env_name, render_mode='human' if render else None)
        rewards = []
        lengths = []

        for _ in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done:
                action, _states = model.predict(state, deterministic=deterministic)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1

            rewards.append(total_reward)
            lengths.append(steps)

        env.close()

        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'rewards': rewards,
            'lengths': lengths
        }

    def evaluate_bc(self, model, num_episodes=10, render=False):
        env = gym.make(self.env_name, render_mode='human' if render else None)
        rewards = []
        lengths = []

        model.eval()

        for _ in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = model.predict(state_tensor)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1

            rewards.append(total_reward)
            lengths.append(steps)

        env.close()

        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'rewards': rewards,
            'lengths': lengths
        }

    def compare_methods(self, dqn_model=None, bc_model=None, num_episodes=10):
        results = {}

        if dqn_model:
            results['dqn'] = self.evaluate_dqn(dqn_model, num_episodes)

        if bc_model:
            results['bc'] = self.evaluate_bc(bc_model, num_episodes)

        return results

    def print_comparison(self, results):
        print("=" * 60)
        print("Method Comparison Results")
        print("=" * 60)

        for method, metrics in results.items():
            print(f"\n{method.upper()}:")
            print(f"  Average Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
            print(f"  Average Length: {metrics['avg_length']:.2f} ± {metrics['std_length']:.2f}")
            print(f"  Max Reward: {metrics['max_reward']:.2f}")
            print(f"  Min Reward: {metrics['min_reward']:.2f}")

        print("\n" + "=" * 60)

        if 'dqn' in results and 'bc' in results:
            dqn_avg = results['dqn']['avg_reward']
            bc_avg = results['bc']['avg_reward']
            improvement = ((dqn_avg - bc_avg) / bc_avg) * 100 if bc_avg != 0 else float('inf')
            
            print(f"\nDQN vs BC Comparison:")
            print(f"  DQN Average Reward: {dqn_avg:.2f}")
            print(f"  BC Average Reward: {bc_avg:.2f}")
            print(f"  DQN Improvement: {improvement:.2f}%")
