# Experiment Summary: protonet_resnet_5way1shot

## Configuration
- **Experiment ID**: 20260706_182421
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
- **Backbone**: resnet18
- **Hidden Size**: 64
- **Embedding Dim**: 512

## Training Configuration
- **Method**: meta
- **Meta LR**: 0.0001
- **Fast LR**: 0.1
- **Epochs**: 100
- **Meta Batch Size**: 1

## Results
- **test_loss**: 1.2313 (±0.3568)
- **test_accuracy**: 0.5680 (±0.1375)
