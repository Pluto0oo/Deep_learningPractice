# Experiment Summary: expA_convnet4

## Configuration
- **Experiment ID**: expA_convnet4
- **Seed**: 42
- **Repeat Times**: 3
- **Device**: cuda

## Data Configuration
- **Dataset**: omniglot
- **Train Ways**: 5
- **Train Shots**: 1
- **Test Ways**: 5
- **Test Shots**: 1

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
- **Meta Batch Size**: 32

## Results
- **test_loss**: 0.6564 (±0.1694)
- **test_accuracy**: 0.7378 (±0.1469)
