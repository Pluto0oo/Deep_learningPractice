"""
优先经验回放（Prioritized Experience Replay, PER）DQN实现

参考论文: Schaul et al., "Prioritized Experience Replay" (ICLR 2016)
论文链接: https://arxiv.org/abs/1511.05952

核心改进:
1. 使用SumTree实现优先级采样，复杂度O(log n)
2. TD-error驱动的事务优先级排序
3. 重要性采样权重(Importance Sampling Weights)修正偏差
"""
import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque


class SumTree:
    """SumTree数据结构，用于高效的事务优先级采样。

    叶节点存储优先级，内部节点存储子节点之和。
    采样复杂度O(log n)，更新复杂度O(log n)。
    """

    def __init__(self, capacity):
        self.capacity = capacity
        # 树节点数 = 2*capacity - 1（叶节点+内部节点）
        self.tree = np.zeros(2 * capacity - 1)
        # 数据数组，存储实际的事务数据
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def _propagate(self, tree_idx, change):
        """向上传播优先级变化"""
        parent = (tree_idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, tree_idx, s):
        """根据累积和s查找对应的叶节点"""
        left = 2 * tree_idx + 1
        right = left + 1

        if left >= len(self.tree):
            return tree_idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """返回根节点（所有优先级之和）"""
        return self.tree[0]

    def add(self, priority, data):
        """添加新事务"""
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        """更新叶节点优先级并传播变化"""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s):
        """根据累积和s采样一个事务"""
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区

    参数:
        capacity: 缓冲区容量
        alpha: 优先级指数（0=均匀采样，1=完全优先级采样）
        beta: 重要性采样权重指数（0=无修正，1=完全修正）
        beta_increment: beta的线性增长步长
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # 防止零优先级
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.max_priority = 1.0

    def _get_priority(self, td_error):
        return (np.abs(td_error) + self.epsilon) ** self.alpha

    def push(self, state, action, reward, next_state, done):
        """添加新事务，初始优先级设为最大值"""
        experience = (state, action, reward, next_state, done)
        priority = self.max_priority
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        """按优先级采样batch_size个事务"""
        states, actions, rewards, next_states, dones = [], [], [], [], []
        indices = []
        weights = []
        priorities = []

        segment = self.tree.total() / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            idx, priority, data = self.tree.get(s)

            # 重要性采样权重: w_i = (N * P(i))^(-beta)
            prob = priority / self.tree.total()
            weight = (self.tree.n_entries * prob) ** (-self.beta)
            weights.append(weight)

            indices.append(idx)
            priorities.append(priority)

            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        # 归一化权重
        weights = np.array(weights)
        weights /= weights.max()

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            weights,
        )

    def update_priorities(self, indices, td_errors):
        """根据新的TD-error更新事务优先级"""
        for idx, td_error in zip(indices, td_errors):
            priority = self._get_priority(td_error)
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries


class QNetwork(nn.Module):
    """Q网络，与SB3 MlpPolicy结构一致"""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class PrioritizedDQNAgent:
    """PER-DQN智能体

    参数:
        state_dim: 状态空间维度
        action_dim: 动作空间维度
        lr: 学习率
        gamma: 折扣因子
        buffer_size: 经验回放缓冲区大小
        batch_size: 批大小
        target_update: 目标网络更新频率
        epsilon_start: 探索率初始值
        epsilon_end: 探索率最小值
        epsilon_decay: 探索率衰减步数
        alpha: PER优先级指数
        beta: PER重要性采样权重指数
    """

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 buffer_size=50000, batch_size=64, target_update=500,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=20000,
                 alpha=0.6, beta=0.4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.action_dim = action_dim

        # epsilon-greedy参数
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay
        self.steps_done = 0

        # Q网络和目标网络
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(buffer_size, alpha=alpha, beta=beta)

    def select_action(self, state):
        """epsilon-greedy动作选择"""
        self.steps_done += 1
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return q_values.argmax().item()

    def train_step(self):
        """执行一步PER-DQN训练"""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # 当前Q值
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 目标Q值: r + gamma * max_a Q_target(s', a)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # TD-error
        td_errors = (current_q - target_q).detach().cpu().numpy()

        # Huber损失，加权重要性采样
        loss = (weights * nn.functional.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # 更新事务优先级
        self.memory.update_priorities(indices, td_errors)

        # 更新目标网络
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()


def train_prioritized_dqn(env_name='CartPole-v1', total_timesteps=50000,
                          eval_interval=5000, eval_episodes=10, seed=42,
                          results_dir='./results'):
    """训练PER-DQN并定期评估

    参数:
        env_name: 环境名称
        total_timesteps: 总训练步数
        eval_interval: 评估间隔
        eval_episodes: 每次评估的回合数
        seed: 随机种子
        results_dir: 结果保存目录
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("=" * 60)
    print("Prioritized Experience Replay DQN")
    print("Paper: Schaul et al., 'Prioritized Experience Replay' (2016)")
    print("=" * 60)
    print(f"Environment: {env_name}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"PER parameters: alpha=0.6, beta=0.4")
    print("=" * 60)

    # 创建智能体
    agent = PrioritizedDQNAgent(state_dim, action_dim, buffer_size=50000)

    # 训练记录
    episode_rewards = []
    episode_lengths = []
    losses = []
    eval_rewards_history = []

    state, _ = env.reset(seed=seed)
    episode_reward = 0
    episode_length = 0
    episode_count = 0

    start_time = time.time()

    for timestep in range(1, total_timesteps + 1):
        # 选择动作
        action = agent.select_action(state)

        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 存入经验回放
        agent.memory.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        episode_length += 1

        # 训练
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)

        # 回合结束
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_count += 1

            if episode_count % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                elapsed = time.time() - start_time
                print(f"Step {timestep}/{total_timesteps} | Episode {episode_count} | "
                      f"Avg Reward (10ep): {avg_reward:.1f} | Epsilon: {agent.epsilon:.3f} | "
                      f"Elapsed: {elapsed:.1f}s")

            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0

        # 定期评估
        if timestep % eval_interval == 0:
            eval_rewards = []
            for _ in range(eval_episodes):
                eval_state, _ = env.reset()
                eval_reward = 0
                eval_done = False
                while not eval_done:
                    with torch.no_grad():
                        state_t = torch.FloatTensor(eval_state).unsqueeze(0).to(agent.device)
                        q_values = agent.q_net(state_t)
                        action = q_values.argmax().item()
                    eval_state, r, term, trunc, _ = env.step(action)
                    eval_reward += r
                    eval_done = term or trunc
                eval_rewards.append(eval_reward)

            eval_avg = np.mean(eval_rewards)
            eval_std = np.std(eval_rewards)
            eval_rewards_history.append({
                'timestep': timestep,
                'avg_reward': eval_avg,
                'std_reward': eval_std,
                'rewards': eval_rewards,
            })
            print(f"\n[EVAL] Step {timestep} | Avg Reward: {eval_avg:.1f} ± {eval_std:.1f}\n")

    # 最终评估
    print("\n" + "=" * 60)
    print("Final Evaluation (10 episodes)")
    print("=" * 60)

    final_rewards = []
    final_lengths = []
    for _ in range(eval_episodes):
        eval_state, _ = env.reset()
        eval_reward = 0
        eval_length = 0
        eval_done = False
        while not eval_done:
            with torch.no_grad():
                state_t = torch.FloatTensor(eval_state).unsqueeze(0).to(agent.device)
                q_values = agent.q_net(state_t)
                action = q_values.argmax().item()
            eval_state, r, term, trunc, _ = env.step(action)
            eval_reward += r
            eval_length += 1
            eval_done = term or trunc
        final_rewards.append(eval_reward)
        final_lengths.append(eval_length)

    final_results = {
        'avg_reward': float(np.mean(final_rewards)),
        'std_reward': float(np.std(final_rewards)),
        'max_reward': float(np.max(final_rewards)),
        'min_reward': float(np.min(final_rewards)),
        'avg_length': float(np.mean(final_lengths)),
        'std_length': float(np.std(final_lengths)),
        'rewards': [float(r) for r in final_rewards],
        'lengths': [int(l) for l in final_lengths],
    }

    print(f"Average Reward: {final_results['avg_reward']:.2f} ± {final_results['std_reward']:.2f}")
    print(f"Max Reward: {final_results['max_reward']:.1f}")
    print(f"Min Reward: {final_results['min_reward']:.1f}")
    print(f"Avg Length: {final_results['avg_length']:.1f}")
    print("=" * 60)

    # 保存结果
    per_results_path = os.path.join(results_dir, 'per_results.npy')
    np.save(per_results_path, final_results)
    print(f"\nResults saved to {per_results_path}")

    # 保存训练历史
    training_history = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'losses': losses,
        'eval_rewards_history': eval_rewards_history,
    }
    history_path = os.path.join(results_dir, 'per_training_history.npy')
    np.save(history_path, training_history, allow_pickle=True)
    print(f"Training history saved to {history_path}")

    # 保存模型
    model_path = os.path.join(results_dir, 'prioritized_dqn_cartpole.pth')
    torch.save({
        'q_net_state_dict': agent.q_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'state_dim': state_dim,
        'action_dim': action_dim,
    }, model_path)
    print(f"Model saved to {model_path}")

    env.close()

    # 保存训练日志
    log_path = os.path.join(results_dir, 'per_training.log')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Prioritized Experience Replay DQN Training Log\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Total timesteps: {total_timesteps}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"PER alpha: 0.6, beta: 0.4\n")
        f.write(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Training Summary:\n")
        f.write(f"  Total episodes: {episode_count}\n")
        f.write(f"  Final epsilon: {agent.epsilon:.4f}\n")
        f.write(f"  Training time: {time.time() - start_time:.1f}s\n")
        f.write(f"  Average loss: {np.mean(losses):.6f}\n\n")
        f.write("Final Evaluation Results:\n")
        f.write(f"  Average reward: {final_results['avg_reward']:.2f} ± {final_results['std_reward']:.2f}\n")
        f.write(f"  Max reward: {final_results['max_reward']:.1f}\n")
        f.write(f"  Min reward: {final_results['min_reward']:.1f}\n")
        f.write(f"  Average length: {final_results['avg_length']:.1f}\n\n")
        f.write("Evaluation History:\n")
        for eval in eval_rewards_history:
            f.write(f"  Step {eval['timestep']}: {eval['avg_reward']:.1f} ± {eval['std_reward']:.1f}\n")
    print(f"Training log saved to {log_path}")

    return final_results, training_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Prioritized DQN with PER")
    parser.add_argument('--config', type=str, default=None, help="Path to config file")
    parser.add_argument('--timesteps', type=int, default=50000, help="Total training timesteps")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()

    results_dir = './results'

    if args.config and os.path.exists(args.config):
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.config import load_config
        config = load_config(args.config)
        results_dir = config.get('results', {}).get('save_dir', './results')
        timesteps = config.get('dqn', {}).get('total_timesteps', 50000)
    else:
        timesteps = args.timesteps

    train_prioritized_dqn(
        total_timesteps=timesteps,
        seed=args.seed,
        results_dir=results_dir,
    )
