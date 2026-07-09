import pytest
import os
from src.config import load_config, merge_configs, generate_exp_id


def test_load_config():
    config = load_config('configs/base.yaml')
    assert 'data' in config
    assert 'model' in config
    assert 'training' in config
    assert 'experiment' in config


def test_merge_configs():
    base = {'a': 1, 'b': {'c': 2, 'd': 3}}
    override = {'a': 10, 'b': {'c': 20}}
    result = merge_configs(base, override)
    assert result['a'] == 10
    assert result['b']['c'] == 20
    assert result['b']['d'] == 3


def test_generate_exp_id():
    config = {
        'experiment': {'algorithm': 'fedavg', 'num_clients': 3, 'num_rounds': 50},
        'model': {'type': 'convnet'},
        'training': {'epochs': 5}
    }
    exp_id = generate_exp_id(config)
    assert 'fedavg' in exp_id
    assert 'convnet' in exp_id
    assert 'clients_3' in exp_id
    assert 'rounds_50' in exp_id