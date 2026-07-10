# Experiment Summary: finetune_5way1shot

## Configuration
- **Experiment ID**: comprehensive_comparison_finetune_5way1shot
- **Seed**: 42
- **Repeat Times**: 1
- **Device**: cuda

## Data Configuration
- **Dataset**: omniglot
- **Train Ways**: 5
- **Train Shots**: 1
- **Test Ways**: 5
- **Test Shots**: 1

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
- **test_loss**: 1.2373
- **test_accuracy**: 0.4933
