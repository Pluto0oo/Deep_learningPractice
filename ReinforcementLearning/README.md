# 强化学习实验项目

本项目在CartPole环境中实现深度Q网络（DQN）和行为克隆（Behavioral Cloning, BC），并包含完整的评估和可视化分析。

## 项目结构

```
ReinforcementLearning/
├── configs/              # 配置文件 (YAML)
├── src/                  # 核心源代码
│   ├── models.py         # 神经网络模型
│   ├── data_collector.py # 专家数据收集
│   ├── trainer.py        # 行为克隆训练器
│   ├── evaluator.py      # 评估模块
│   ├── visualization.py  # 结果可视化
│   └── config.py         # 配置加载器
├── scripts/              # 执行脚本
│   ├── train_dqn.py      # 训练DQN智能体
│   ├── collect_expert_data.py # 收集专家示范数据
│   ├── train_bc.py       # 训练行为克隆模型
│   ├── evaluate.py       # 评估和对比模型
│   ├── train_prioritized_dqn.py # 优先经验回放
│   └── run_full_experiment.py   # 运行完整实验流程
├── results/              # 实验结果
│   ├── training.log      # 训练日志
│   ├── evaluation_results.npy # 评估结果数据
│   ├── bc_accuracy_history.npy # BC训练准确率历史
│   ├── bc_loss_history.npy     # BC训练损失历史
│   ├── dqn_cartpole.zip  # DQN模型
│   ├── bc_cartpole.pth   # BC模型
│   └── plots/            # 可视化图表
│       ├── reward_comparison.png   # 奖励对比图
│       ├── reward_distribution.png # 奖励分布图
│       └── confidence_interval.png # 置信区间图
├── data/                 # 专家数据存储
├── docs/                 # 文档
│   ├── experiment_principle.md  # 实验原理与结果分析
│   └── data_analysis.md         # 数据分析报告
├── requirements.txt      # 依赖
└── .gitignore            # Git忽略规则
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

4. **评估模型**
```bash
python scripts/evaluate.py --config configs/base.yaml
```

### 论文复现实验

**优先经验回放（Prioritized Experience Replay, PER）**
```bash
python scripts/train_prioritized_dqn.py
```

## 实验结果

### 评估结果对比（10轮评估平均值）

| 方法 | 平均奖励 | 标准差 | 最大奖励 | 最小奖励 |
|------|----------|--------|----------|----------|
| DQN | **107.3** | 2.79 | 112.0 | 103.0 |
| BC | 106.7 | 3.16 | **113.0** | 103.0 |

### BC训练过程

| 指标 | 初始值 | 最终值 | 最优值 |
|------|--------|--------|--------|
| 训练准确率 | 0.8146 | 0.9677 | **0.9684** (Epoch 97) |
| 训练损失 | 0.4155 | 0.0788 | **0.0771** (Epoch 98) |

### 关键发现

1. **DQN与BC性能接近**：平均奖励仅差0.6（107.3 vs 106.7），表明BC在简单环境中可媲美DQN
2. **BC训练收敛稳定**：100轮内准确率从81.5%提升至96.8%，损失降至0.077
3. **BC波动略大**：标准差3.16 > DQN的2.79，反映分布偏移的影响
4. **两种方法均未达最大值500**：可能因训练步数不足或超参数需调优

详细数据分析请参阅 [docs/data_analysis.md](docs/data_analysis.md)

## 依赖

- stable-baselines3[extra] == 2.0.0
- gymnasium == 0.29.1
- numpy == 1.26.4
- pandas == 2.2.1
- matplotlib == 3.8.4
- seaborn == 0.13.2
- scipy == 1.12.0
- pyyaml == 6.0.1
- tqdm == 4.66.2
- pytest == 8.2.0
- scikit-learn == 1.4.1

## 实验说明

### 深度Q网络（DQN）
- 使用MlpPolicy，2个隐藏层
- 经验回放缓冲区（容量1M）
- 目标网络每10,000步更新
- 训练频率：每4步
- 总训练步数：1,000,000

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

## 参考文献

1. Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning[J]. Nature, 2015, 518(7540): 529-533.

2. Schaul T, Quan J, Antonoglou I, et al. Prioritized experience replay[J]. arXiv preprint arXiv:1511.05952, 2015.

3. Ho J, Ermon S. Generative adversarial imitation learning[C]//Advances in neural information processing systems. 2016: 4565-4573.

4. Levine S, Finn C, Darrell T, et al. End-to-end training of deep visuomotor policies[J]. Journal of Machine Learning Research, 2016, 17(1): 1334-1373.

## 许可证

MIT License
