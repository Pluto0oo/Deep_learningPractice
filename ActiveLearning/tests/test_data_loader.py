import os
import sys
import pytest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import create_sample_data, get_data_stats, load_data


def test_create_sample_data(tmp_path):
    data_path = str(tmp_path / "data")
    
    create_sample_data(data_path, n_samples=100, n_features=10, n_classes=3)
    
    assert os.path.exists(os.path.join(data_path, "X.npy"))
    assert os.path.exists(os.path.join(data_path, "y.npy"))
    
    X = np.load(os.path.join(data_path, "X.npy"))
    y = np.load(os.path.join(data_path, "y.npy"))
    
    assert X.shape == (100, 10)
    assert y.shape == (100,)
    assert len(np.unique(y)) == 3


def test_get_data_stats():
    data = {
        "X": np.random.randn(100, 20),
        "y": np.array([0] * 20 + [1] * 30 + [2] * 50),
        "X_val": np.random.randn(20, 20),
        "y_val": np.array([0] * 5 + [1] * 7 + [2] * 8),
    }
    
    stats = get_data_stats(data)
    
    assert stats["n_samples"] == 100
    assert stats["n_features"] == 20
    assert stats["n_classes"] == 3
    assert stats["class_distribution"] == {0: 20, 1: 30, 2: 50}
    assert stats["n_val_samples"] == 20