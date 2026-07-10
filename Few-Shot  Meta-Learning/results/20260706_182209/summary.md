# Experiment Summary: protonet_convnet6_5way1shot

## Configuration
- **Experiment ID**: 20260706_182209
- **Seed**: 42
- **Repeat Times**: 5
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
- **Meta Batch Size**: 1

## Results
- **test_loss**: 0.6535 (±0.2276)
- **test_accuracy**: 0.7547 (±0.1365)
