# Experiment Summary: protonet_5way5shot

## Configuration
- **Experiment ID**: comprehensive_comparison_protonet_5way5shot
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
- **Model Type**: protonet
- **Backbone**: convnet
- **Hidden Size**: 64
- **Embedding Dim**: 64

## Training Configuration
- **Method**: meta
- **Meta LR**: 0.001
- **Fast LR**: 0.4
- **Epochs**: 100
- **Meta Batch Size**: 1

## Results
- **test_loss**: 0.5376 (±0.1918)
- **test_accuracy**: 0.7973 (±0.1266)
