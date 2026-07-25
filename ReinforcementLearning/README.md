# Reinforcement Learning Project

This project implements Deep Q-Network (DQN) and Behavioral Cloning for the CartPole environment, with comprehensive evaluation and visualization.

## Project Structure

```
ReinforcementLearning/
├── configs/              # Configuration files (YAML)
├── src/                  # Core source code
│   ├── models.py         # Neural network models
│   ├── data_collector.py # Expert data collection
│   ├── trainer.py        # Behavioral Cloning trainer
│   ├── evaluator.py      # Evaluation module
│   ├── visualization.py  # Result visualization
│   └── config.py         # Configuration loader
├── scripts/              # Execution scripts
│   ├── train_dqn.py      # Train DQN agent
│   ├── collect_expert_data.py # Collect expert demonstrations
│   ├── train_bc.py       # Train Behavioral Cloning model
│   ├── evaluate.py       # Evaluate and compare models
│   ├── train_prioritized_dqn.py # Prioritized Experience Replay
│   └── run_full_experiment.py   # Run full experiment pipeline
├── results/              # Experiment results
├── data/                 # Expert data storage
├── docs/                 # Documentation
├── requirements.txt      # Dependencies
└── .gitignore            # Git ignore rules
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Run Full Experiment Pipeline

```bash
python scripts/run_full_experiment.py --config configs/base.yaml
```

### Step-by-Step Execution

1. **Train DQN Agent**
```bash
python scripts/train_dqn.py --config configs/base.yaml
```

2. **Collect Expert Data**
```bash
python scripts/collect_expert_data.py --config configs/base.yaml
```

3. **Train Behavioral Cloning**
```bash
python scripts/train_bc.py --config configs/base.yaml
```

4. **Evaluate Models**
```bash
python scripts/evaluate.py --config configs/base.yaml
```

### Paper Reproduction Experiments

**Prioritized Experience Replay (PER)**
```bash
python scripts/train_prioritized_dqn.py
```

## Dependencies

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

## Experiment Description

### Deep Q-Network (DQN)
- Uses MlpPolicy with 2 hidden layers
- Experience replay buffer (1M capacity)
- Target network updates every 10,000 steps
- Training frequency: every 4 steps
- Total training: 1M timesteps

### Behavioral Cloning (BC)
- Supervised learning approach
- Trained on expert demonstrations from DQN
- 2-layer MLP with 128 hidden units
- Cross-entropy loss
- 100 training epochs

### Evaluation Metrics
- Average reward over 10 episodes
- Standard deviation of rewards
- Average episode length
- 95% confidence intervals

## Paper References

1. Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning[J]. Nature, 2015, 518(7540): 529-533.

2. Schaul T, Quan J, Antonoglou I, et al. Prioritized experience replay[J]. arXiv preprint arXiv:1511.05952, 2015.

3. Ho J, Ermon S. Generative adversarial imitation learning[C]//Advances in neural information processing systems. 2016: 4565-4573.

4. Levine S, Finn C, Darrell T, et al. End-to-end training of deep visuomotor policies[J]. Journal of Machine Learning Research, 2016, 17(1): 1334-1373.

## License

MIT License
