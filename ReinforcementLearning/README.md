# 强化学习实验项目

本项目在CartPole环境中实现深度Q网络（DQN）、优先经验回放（PER-DQN）和行为克隆（Behavioral Cloning, BC），并包含完整的评估和可视化分析。

## 项目结构

```
ReinforcementLearning/
├── configs/                        # 配置文件 (YAML)
├── src/                            # 核心源代码
│   ├── models.py                   # 神经网络模型
│   ├── data_collector.py           # 专家数据收集
│   ├── trainer.py                  # 行为克隆训练器
│   ├── evaluator.py                # 评估模块
│   ├── visualization.py           # 结果可视化（改进版）
│   └── config.py                   # 配置加载器
├── scripts/                        # 执行脚本
│   ├── train_dqn.py                # 训练DQN智能体
│   ├── collect_expert_data.py       # 收集专家示范数据
│   ├── train_bc.py                 # 训练行为克隆模型
│   ├── evaluate.py                 # 评估和对比模型（含PER）
│   ├── train_prioritized_dqn.py    # 优先经验回放PER-DQN（完整实现）
│   └── run_full_experiment.py       # 运行完整实验流程
├── results/                        # 实验结果
│   ├── training.log                # DQN训练日志
│   ├── per_training.log            # PER-DQN训练日志
│   ├── evaluation_results.npy      # 评估结果数据
│   ├── per_results.npy             # PER-DQN评估结果
│   ├── per_training_history.npy    # PER-DQN训练历史
│   ├── bc_accuracy_history.npy     # BC训练准确率历史
│   ├── bc_loss_history.npy         # BC训练损失历史
│   ├── dqn_cartpole.zip            # DQN模型
│   ├── bc_cartpole.pth             # BC模型
│   ├── prioritized_dqn_cartpole.pth # PER-DQN模型
│   └── plots/                      # 可视化图表
│       ├── reward_comparison.png       # 奖励对比图（改进版）
│       ├── reward_distribution.png     # 奖励分布箱线图
│       ├── confidence_interval.png     # 95%置信区间图
│       ├── comprehensive_comparison.png # 综合对比面板（4子图）
│       ├── bc_training.png             # BC训练曲线
│       └── per_training_curve.png      # PER-DQN训练曲线
├── data/                           # 专家数据存储
├── docs/                           # 文档
│   ├── experiment_principle.md     # 实验原理与结果分析
│   └── data_analysis.md            # 数据分析报告
├── requirements.txt                # 依赖
└── .gitignore                      # Git忽略规则
```

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 运行完整实验流程

```bash
python scripts/run_full_experiment.py --config configs/base.yaml
```

### 分步执行

1. **训练DQN智能体**
```bash
python scripts/train_dqn.py --config configs/base.yaml
```

2. **收集专家数据**
```bash
python scripts/collect_expert_data.py --config configs/base.yaml
```

3. **训练行为克隆模型**
```bash
python scripts/train_bc.py --config configs/base.yaml
```

4. **训练优先经验回放DQN（PER-DQN）**
```bash
python scripts/train_prioritized_dqn.py --timesteps 50000 --seed 42
```

5. **评估所有模型并生成可视化**
```bash
python scripts/evaluate.py --config configs/base.yaml
```

## 实验结果

### 评估结果对比（10轮评估平均值）

| 方法 | 平均奖励 | 标准差 | 最大奖励 | 最小奖励 | 变异系数 |
|------|----------|--------|----------|----------|----------|
| DQN | 107.3 | 2.79 | 112.0 | 103.0 | 2.60% |
| BC | 106.7 | 3.16 | **113.0** | 103.0 | 2.96% |
| **PER-DQN** | **210.2** | 7.28 | **222.0** | **194.0** | 3.46% |

> PER-DQN相比标准DQN提升95.8%，平均奖励翻倍

### PER-DQN训练过程

| 指标 | 值 |
|------|-----|
| 总训练步数 | 50,000 |
| 总回合数 | 952 |
| 训练时间 | 448.2秒 |
| 最终ε | 0.050 |
| 平均损失 | 0.0687 |
| 设备 | CUDA |

### BC训练过程

| 指标 | 初始值 | 最终值 | 最优值 |
|------|--------|--------|--------|
| 训练准确率 | 0.8146 | 0.9677 | **0.9684** (Epoch 97) |
| 训练损失 | 0.4155 | 0.0788 | **0.0771** (Epoch 98) |

### 关键发现

1. **PER-DQN显著优于标准DQN**：平均奖励210.2 vs 107.3，提升95.8%
2. **PER的优先级采样有效**：高TD-error经验被更频繁采样，加速学习
3. **DQN与BC性能接近**：差异仅0.6，表明BC在简单环境中可媲美DQN
4. **BC训练收敛稳定**：100轮内准确率从81.5%提升至96.8%
5. **所有方法均未达最大值500**：需进一步调优超参数或增加训练步数

### 可视化图表

#### 奖励对比图
![奖励对比](results/plots/reward_comparison.png)

#### 综合对比面板
![综合对比](results/plots/comprehensive_comparison.png)

#### PER-DQN训练曲线
![PER训练曲线](results/plots/per_training_curve.png)

#### 奖励分布
![奖励分布](results/plots/reward_distribution.png)

#### 95%置信区间
![置信区间](results/plots/confidence_interval.png)

详细数据分析请参阅 [docs/data_analysis.md](docs/data_analysis.md)

## 实验说明

### 深度Q网络（DQN）
- 使用MlpPolicy，2个隐藏层
- 经验回放缓冲区（容量1M）
- 目标网络每10,000步更新
- 训练频率：每4步
- 总训练步数：1,000,000

### 优先经验回放DQN（PER-DQN）
- **论文参考**：Schaul et al., "Prioritized Experience Replay" (ICLR 2016)
- **核心改进**：SumTree数据结构实现O(log n)优先级采样
- **优先级参数**：α=0.6（优先级指数），β=0.4→1.0（重要性采样权重）
- **网络结构**：2层MLP（64维隐藏层），与DQN一致
- **经验回放**：容量50,000，batch_size=64
- **探索策略**：ε-greedy，ε从1.0线性衰减至0.05
- **总训练步数**：50,000

#### PER核心算法

**采样概率**（按TD-error优先级）：
$$P(i) = \frac{|\text{TD-error}_i|^\alpha}{\sum_j |\text{TD-error}_j|^\alpha}$$

**重要性采样权重**（修正偏差）：
$$w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$$

### 行为克隆（BC）
- 监督学习方法
- 基于DQN的专家示范训练
- 2层MLP，128个隐藏单元
- 交叉熵损失
- 100个训练轮次

### 评估指标
- 10轮平均奖励
- 奖励标准差
- 平均回合长度
- 95%置信区间
- 变异系数（CV）

## 参数配置指南

### DQN参数

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| learning_rate | 0.001 | 0.0001-0.01 | 学习率 |
| buffer_size | 1,000,000 | 50,000-1,000,000 | 经验回放容量 |
| batch_size | 64 | 32-256 | 批大小 |
| gamma | 0.99 | 0.9-0.999 | 折扣因子 |
| target_update_interval | 10,000 | 1,000-50,000 | 目标网络更新频率 |
| total_timesteps | 1,000,000 | 50,000-1,000,000 | 总训练步数 |

### PER-DQN参数

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| alpha (α) | 0.6 | 0-1 | 优先级指数（0=均匀，1=完全优先） |
| beta (β) | 0.4 | 0-1 | 重要性采样权重（0=无修正） |
| beta_increment | 0.001 | 0.0001-0.01 | β线性增长步长 |
| epsilon (ε) | 1e-6 | 1e-8-1e-4 | 防止零优先级 |
| buffer_size | 50,000 | 10,000-500,000 | PER缓冲区容量 |
| epsilon_greedy_start | 1.0 | 0.5-1.0 | 探索率初始值 |
| epsilon_greedy_end | 0.05 | 0.01-0.1 | 探索率最小值 |

## 常见问题

### Q1: PER-DQN训练为什么比DQN慢？
PER使用SumTree数据结构，每次采样和更新优先级都需要O(log n)操作，比均匀采样的O(1)更耗时。但采样质量更高，总体学习效率更优。

### Q2: alpha和beta参数如何调节？
- **alpha**：控制优先级程度。alpha=0退化为均匀采样，alpha=1为完全优先级采样。通常0.4-0.7效果较好。
- **beta**：修正非均匀采样引入的偏差。beta从0.4线性增长到1.0，训练后期偏差修正更完全。

### Q3: 为什么DQN和BC的性能如此接近？
在CartPole-v1这种简单环境中，BC通过模仿专家即可达到接近DQN的性能。但在更复杂的环境中，BC的分布偏移问题会更明显。

### Q4: 为什么所有方法都未达到最大值500？
可能原因：1) 训练步数不足；2) 超参数需调优；3) 网络架构过于简单；4) 需要更长的探索阶段。

### Q5: 如何选择DQN还是PER-DQN？
- **简单环境**：DQN足够，训练更快
- **复杂环境**：PER-DQN更优，优先级采样加速学习
- **稀疏奖励**：PER-DQN优势更明显，高TD-error经验更重要

## 参考文献

1. Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning[J]. Nature, 2015, 518(7540): 529-533.

2. Schaul T, Quan J, Antonoglou I, et al. Prioritized experience replay[J]. arXiv preprint arXiv:1511.05952, 2015.

3. Ho J, Ermon S. Generative adversarial imitation learning[C]//Advances in neural information processing systems. 2016: 4565-4573.

4. Levine S, Finn C, Darrell T, et al. End-to-end training of deep visuomotor policies[J]. Journal of Machine Learning Research, 2016, 17(1): 1334-1373.

## 版本信息

- **版本**: v2.0.0
- **更新日期**: 2026-07-31
- **主要变更**: 
  - 新增PER-DQN完整实现（SumTree+重要性采样）
  - 改进可视化模块（6种图表+专业配色）
  - 修复.gitignore使结果文件可上传GitHub
  - 完善文档与参数配置指南

## 许可证

MIT License
