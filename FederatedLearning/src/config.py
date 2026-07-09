import os
import yaml
from typing import Dict


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    base_config_path = config.get('base', None)
    if base_config_path:
        base_config = load_config(base_config_path)
        config = merge_configs(base_config, config)

    return config


def merge_configs(base: Dict, override: Dict) -> Dict:
    result = base.copy()
    for key, value in override.items():
        if key == 'base':
            continue
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def generate_exp_id(config: Dict) -> str:
    parts = []
    parts.append(config['experiment'].get('algorithm', 'fedavg'))
    parts.append(config['model'].get('type', 'convnet'))
    parts.append(f"clients_{config['experiment'].get('num_clients', 3)}")
    parts.append(f"rounds_{config['experiment'].get('num_rounds', 50)}")
    parts.append(f"epochs_{config['training'].get('epochs', 5)}")
    return '_'.join(parts)