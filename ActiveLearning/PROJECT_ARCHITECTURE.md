# ActiveLearning Project Architecture

## Overview

This document describes the project architecture and rules for the ActiveLearning experiment management framework.

## Directory Structure

```
ActiveLearning/
├── configs/          # Configuration files (YAML)
├── data/             # Data directory
│   ├── raw/          # Raw data
│   └── processed/    # Processed data
├── src/              # Source code
├── scripts/          # Scripts for running experiments
├── notebooks/        # Jupyter notebooks for analysis
├── results/          # Experiment results (by exp_id)
├── logs/             # Log files (by exp_id)
├── reports/          # Generated reports
└── tests/            # Test files
```

## Experiment Identification

- Each experiment is identified by a unique `exp_id`
- The `exp_id` can be specified via `--exp_id` parameter or auto-generated as `{name}_{timestamp}`
- Auto-generated format: `{name}_{YYYYMMDD}_{HHMMSS}`

## Configuration System

### Format

All configurations use YAML format with the following structure:

```yaml
experiment:
  name: str            # Experiment name
  seed: int            # Random seed for reproducibility
  repeat_times: int    # Number of times to repeat the experiment

data:
  path: str            # Path to processed data
  test_size: float     # Proportion of data for testing (0.0-1.0)
  val_size: float      # Proportion of training data for validation (0.0-1.0)

model:
  type: str            # Model type (e.g., SimpleMLP)
  input_dim: int       # Input dimension
  hidden_dim: int      # Hidden layer dimension
  num_layers: int      # Number of hidden layers
  output_dim: int      # Output dimension (number of classes)
  dropout_rate: float  # Dropout rate (0.0-1.0)

training:
  epochs: int          # Number of training epochs
  batch_size: int      # Batch size
  loss: str            # Loss function (cross_entropy, mse)
  optimizer: str       # Optimizer (adam, sgd, adamw)
  learning_rate: float # Learning rate
  weight_decay: float  # Weight decay for regularization
  scheduler: str       # Learning rate scheduler (None, step, cosine, reduce_on_plateau)
  save_best: bool      # Whether to save best model

evaluation:
  metrics: list        # Metrics to compute (accuracy, precision, recall, f1)
```

### Example Configurations

- `configs/base.yaml`: Base configuration with default values
- `configs/example_experiment.yaml`: Example experiment configuration

## Result Storage

### Structure

Results are stored in `results/exp_id/`:

```
results/exp_id/
├── config_used.yaml   # Configuration used for the experiment
├── metrics.csv        # Metrics for each epoch (CSV format)
├── metrics.json       # Final metrics (JSON format)
├── summary.md         # Auto-generated summary (Markdown)
├── plots/             # Generated plots (PNG)
└── checkpoints/       # Model checkpoints (PyTorch)
    ├── best_model.pt  # Best model by validation accuracy
    └── final_model.pt # Final model after all epochs
```

### Multiple Repeats

If `repeat_times > 1`:

```
results/exp_id/
└── repeats/
    ├── repeat_000/    # Results for repeat 0
    ├── repeat_001/    # Results for repeat 1
    ├── ...
    └── stats/         # Statistical summary across all repeats
        └── stats.json # Mean, std, min, max for each metric
```

## Logging

- Logs are stored in `logs/exp_id.log`
- Uses Python's `logging` module
- Format: `{timestamp} - {logger_name} - {level} - {message}`
- Logs are written to both file and console

## Scripts

### run_experiment.py

Run a single experiment.

**Parameters:**
- `--config`: Path to config file (required)
- `--exp_id`: Experiment ID (optional, auto-generated if not provided)

**Features:**
- Loads configuration
- Sets random seed
- Creates experiment directories
- Runs training with support for multiple repeats
- Saves metrics, checkpoints, and summary

### run_comparison.py

Run multiple experiments and generate a comparison report.

**Parameters:**
- `--configs`: List of config files to compare (required)
- `--output`: Output path for comparison report (default: reports/comparison_report.md)

### aggregate_results.py

Aggregate results from all experiments.

**Parameters:**
- `--results_dir`: Directory containing experiment results (default: results)
- `--output_dir`: Output directory for aggregated results (default: reports)

**Output:**
- `aggregated_results.csv`: CSV file with all experiment metrics
- `aggregated_results.xlsx`: Excel file with all experiment metrics

### generate_report.py

Generate a comprehensive report from all experiments.

**Parameters:**
- `--results_dir`: Directory containing experiment results (default: results)
- `--output`: Output path for the report (default: reports/final_report.md)

### analyze_results.py

Analyze and visualize experiment results.

**Parameters:**
- `--results_dir`: Directory containing experiment results (default: results)
- `--output_dir`: Output directory for plots (default: reports/plots)

## Core Modules

### src/utils.py

- `set_seed(seed)`: Set random seeds for reproducibility
- `setup_logger(exp_id, log_dir)`: Create logger for experiment
- `load_config(config_path)`: Load YAML configuration file
- `save_config(config, save_path)`: Save configuration to YAML file
- `generate_exp_id(name)`: Generate experiment ID with timestamp
- `create_experiment_dirs(exp_id, results_dir)`: Create directories for experiment
- `get_device()`: Get available device (GPU or CPU)
- `save_metrics(metrics, save_path)`: Save metrics to YAML file

### src/data_loader.py

- `load_data(config)`: Load and split data into train/val/test
- `create_sample_data(data_path, n_samples, n_features, n_classes)`: Create sample data for testing
- `get_data_stats(data)`: Get statistics about the data

### src/model.py

- `BaseModel`: Base class for all models
- `SimpleMLP`: Simple multi-layer perceptron model
- `build_model(config)`: Build model based on configuration
- `get_loss_function(config)`: Get loss function based on configuration
- `get_optimizer(config, model)`: Get optimizer based on configuration
- `get_scheduler(config, optimizer)`: Get learning rate scheduler based on configuration

### src/trainer.py

- `train_model(config, train_data, device, logger, exp_dirs)`: Train model and return final metrics

### src/evaluator.py

- `evaluate_model(model, data, criterion, device)`: Evaluate model and return metrics
- `compute_metrics(y_true, y_pred)`: Compute classification metrics
- `predict_model(model, X, device)`: Make predictions using model

## Rules

1. **Reproducibility**: All experiments must use fixed random seeds
2. **Configuration**: All parameters must be specified in YAML config files
3. **Result Organization**: Results must be organized by experiment ID
4. **Logging**: All experiments must log to both file and console
5. **Error Handling**: All scripts must include proper error handling
6. **Parameter Parsing**: All scripts must use argparse for command-line arguments
7. **Code Quality**: All code must follow PEP 8 standards

## Testing

- Test files are located in `tests/`
- Run tests with `pytest tests/ -v`

## Dependencies

See `requirements.txt` for a complete list of dependencies.

## Versioning

- Use semantic versioning
- Update version in README.md and setup.py (if applicable)