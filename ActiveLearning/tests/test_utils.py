import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils import (
    set_seed,
    generate_exp_id,
    create_experiment_dirs,
    load_config,
    save_config,
)


def test_set_seed():
    set_seed(42)
    
    import random
    import numpy as np
    
    assert random.randint(0, 100) == random.randint(0, 100)
    
    np.random.seed(42)
    expected = np.random.rand(5)
    set_seed(42)
    actual = np.random.rand(5)
    
    np.testing.assert_array_equal(expected, actual)


def test_generate_exp_id():
    exp_id = generate_exp_id("test")
    
    assert isinstance(exp_id, str)
    assert exp_id.startswith("test_")
    assert len(exp_id) == len("test_") + len("20260630_232200")


def test_create_experiment_dirs():
    exp_id = "test_exp"
    
    dirs = create_experiment_dirs(exp_id, "results/test")
    
    assert "base" in dirs
    assert "plots" in dirs
    assert "checkpoints" in dirs
    assert "repeats" in dirs
    
    for dir_path in dirs.values():
        assert os.path.exists(dir_path)
    
    import shutil
    shutil.rmtree("results/test")


def test_config_io(tmp_path):
    config = {
        "experiment": {"name": "test", "seed": 42},
        "model": {"type": "SimpleMLP"},
    }
    
    config_path = str(tmp_path / "test_config.yaml")
    
    save_config(config, config_path)
    assert os.path.exists(config_path)
    
    loaded_config = load_config(config_path)
    assert loaded_config == config