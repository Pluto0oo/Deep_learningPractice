import os
import sys
import logging
import random
import numpy as np
import torch
import yaml
from datetime import datetime
from typing import Dict, Any, Optional


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(exp_id: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{exp_id}.log")
    
    logger = logging.getLogger(exp_id)
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    stream_handler.flush = sys.stdout.flush
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def generate_exp_id(name: str = "exp") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{timestamp}"


def create_experiment_dirs(exp_id: str, results_dir: str = "results") -> Dict[str, str]:
    base_dir = os.path.join(results_dir, exp_id)
    dirs = {
        "base": base_dir,
        "plots": os.path.join(base_dir, "plots"),
        "checkpoints": os.path.join(base_dir, "checkpoints"),
        "repeats": os.path.join(base_dir, "repeats"),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_metrics(metrics: Dict[str, Any], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)


def load_metrics(metrics_path: str) -> Dict[str, Any]:
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    with open(metrics_path, "r") as f:
        metrics = yaml.safe_load(f)
    
    return metrics