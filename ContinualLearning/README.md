# Continual Learning Project

This project implements and compares various continual learning methods for text classification tasks.

## Project Structure

```
ContinualLearning/
├── configs/
│   └── base.yaml          # Configuration file
├── docs/
│   └── experiment_principle.md  # Experiment documentation
├── scripts/
│   ├── run_fine_tuning.py # Fine tuning baseline
│   ├── run_ewc.py         # Elastic Weight Consolidation
│   ├── run_lwf.py         # Learning without Forgetting
│   ├── run_der.py         # Dark Experience Replay
│   └── run_comparison.py  # Compare all methods
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration loader
│   ├── data.py            # Data loading and experience buffer
│   ├── models.py          # Model definitions
│   ├── trainer.py         # Training classes
│   ├── evaluator.py       # Evaluation metrics
│   └── visualization.py   # Result visualization
├── results/               # Experiment results
├── data/                  # Dataset storage
├── requirements.txt       # Dependencies
└── README.md
```

## Installation

```bash
conda create -n dlp python=3.9
conda activate dlp
pip install -r requirements.txt
```

## Usage

### Run Fine Tuning Baseline

```bash
python scripts/run_fine_tuning.py --config configs/base.yaml
```

### Run EWC (Elastic Weight Consolidation)

```bash
python scripts/run_ewc.py --config configs/base.yaml
```

### Run LWF (Learning without Forgetting)

```bash
python scripts/run_lwf.py --config configs/base.yaml
```

### Run DER (Dark Experience Replay)

```bash
python scripts/run_der.py --config configs/base.yaml
```

### Compare All Methods

```bash
python scripts/run_comparison.py --config configs/base.yaml
```

## Experiments

### Task Setup
- **Task 1**: IMDB Sentiment Classification (2 classes)
- **Task 2**: AG News Topic Classification (4 classes)

### Methods Compared
1. **Fine Tuning**: Directly fine-tune the model on sequential tasks
2. **EWC**: Elastic Weight Consolidation - protects important weights
3. **LWF**: Learning without Forgetting - knowledge distillation from old model
4. **DER**: Dark Experience Replay - combines replay buffer with knowledge distillation

### Evaluation Metrics
- Accuracy on each task after training
- Forgetting rate
- Training time
- Memory usage

## Results

Results are saved to `results/` directory including:
- Training curves
- Task accuracy plots
- Forgetting rate comparison
- Runtime comparison
- Memory usage comparison

## References

1. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS.
2. Li, Z., & Hoiem, D. (2017). Learning without forgetting. ECCV.
3. Buzzega, P., et al. (2020). Dark experience for general continual learning: A strong, simple baseline. NeurIPS.