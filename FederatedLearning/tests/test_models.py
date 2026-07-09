import pytest
import torch
from src.models import ConvNet, ResNet18, create_model


def test_convnet_output_shape():
    model = ConvNet(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    assert output.shape == (2, 10)


def test_resnet18_output_shape():
    model = ResNet18(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    assert output.shape == (2, 10)


def test_create_model_convnet():
    config = {'model': {'type': 'convnet'}, 'data': {'num_classes': 10}}
    model = create_model(config)
    assert isinstance(model, ConvNet)


def test_create_model_resnet():
    config = {'model': {'type': 'resnet18'}, 'data': {'num_classes': 10}}
    model = create_model(config)
    assert isinstance(model, ResNet18)


def test_model_device():
    model = ConvNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    assert next(model.parameters()).device == device