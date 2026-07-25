import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN


class ExpertDataCollector:
    def __init__(self, env_name='CartPole-v1', model_path=None):
        self.env_name = env_name
        self.model_path = model_path
        self.env = gym.make(env_name)
        self.model = None
        self.expert_data = []

    def load_expert_model(self):
        if self.model_path and os.path.exists(self.model_path):
            self.model = DQN.load(self.model_path)
            return True
        return False

    def collect_episode(self, max_steps=500):
        if self.model is None:
            raise ValueError("Expert model not loaded")

        state, _ = self.env.reset()
        episode_data = []
        done = False
        steps = 0

        while not done and steps < max_steps:
            action, _states = self.model.predict(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            episode_data.append({
                'state': state,
                'action': action,
                'next_state': next_state,
                'reward': reward,
                'done': done
            })
            
            state = next_state
            steps += 1

        return episode_data, steps

    def collect_data(self, num_episodes=100, save_path=None):
        self.expert_data = []
        total_steps = 0

        for i in range(num_episodes):
            episode_data, steps = self.collect_episode()
            self.expert_data.extend(episode_data)
            total_steps += steps

            if (i + 1) % 10 == 0:
                print(f"Collected {i + 1}/{num_episodes} episodes, total steps: {total_steps}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, self.expert_data)
            print(f"Expert data saved to {save_path}")

        return self.expert_data

    def get_statistics(self):
        if not self.expert_data:
            return {}

        rewards = [np.sum([step['reward'] for step in episode_data]) 
                   for episode_data in self._split_episodes()]
        
        return {
            'num_episodes': len(self._split_episodes()),
            'total_samples': len(self.expert_data),
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards)
        }

    def _split_episodes(self):
        episodes = []
        current_episode = []
        
        for step in self.expert_data:
            current_episode.append(step)
            if step['done']:
                episodes.append(current_episode)
                current_episode = []
        
        if current_episode:
            episodes.append(current_episode)
        
        return episodes

    def close(self):
        self.env.close()
