# Experiment Summary: finetune_5way5shot

## Configuration
- **Experiment ID**: comprehensive_comparison_finetune_5way5shot
- **Seed**: 42
- **Repeat Times**: 5
- **Device**: cuda

## Data Configuration
- **Dataset**: omniglot
- **Train Ways**: 5
- **Train Shots**: 5
- **Test Ways**: 5
- **Test Shots**: 5

## Model Configuration
- **Model Type**: finetune
- **Backbone**: convnet
- **Hidden Size**: 64
- **Embedding Dim**: 64

## Training Configuration
- **Method**: finetune
- **Meta LR**: 0.001
- **Fast LR**: 0.1
- **Epochs**: 100
- **Meta Batch Size**: 1

## Results
- **test_loss**: 1.0204 (±0.1657)
- **test_accuracy**: 0.5893 (±0.1047)
