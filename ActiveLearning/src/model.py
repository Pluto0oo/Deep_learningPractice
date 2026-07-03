import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class BaseModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(BaseModel, self).__init__()
        self.config = config
        self.model_config = config.get("model", {})
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward method must be implemented")
    
    def save_checkpoint(self, save_path: str, epoch: int, metrics: Dict[str, float] = None) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "config": self.config,
        }
        if metrics:
            checkpoint["metrics"] = metrics
        
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class SimpleMLP(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super(SimpleMLP, self).__init__(config)
        
        input_dim = self.model_config.get("input_dim", 20)
        hidden_dim = self.model_config.get("hidden_dim", 128)
        num_layers = self.model_config.get("num_layers", 2)
        output_dim = self.model_config.get("output_dim", 10)
        dropout_rate = self.model_config.get("dropout_rate", 0.5)
        
        layers = []
        prev_dim = input_dim
        
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_model(config: Dict[str, Any]) -> BaseModel:
    model_type = config["model"].get("type", "SimpleMLP")
    
    if model_type == "SimpleMLP":
        return SimpleMLP(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_loss_function(config: Dict[str, Any]) -> nn.Module:
    loss_type = config["training"].get("loss", "cross_entropy")
    
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_type == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_optimizer(config: Dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    optimizer_type = config["training"].get("optimizer", "adam")
    lr = config["training"].get("learning_rate", 0.001)
    weight_decay = config["training"].get("weight_decay", 0.0)
    
    params = model.parameters()
    
    if optimizer_type == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        momentum = config["training"].get("momentum", 0.9)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_scheduler(config: Dict[str, Any], optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    scheduler_type = config["training"].get("scheduler", None)
    
    if scheduler_type is None:
        return None
    
    if scheduler_type == "step":
        step_size = config["training"].get("step_size", 10)
        gamma = config["training"].get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "cosine":
        T_max = config["training"].get("epochs", 100)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == "reduce_on_plateau":
        factor = config["training"].get("factor", 0.1)
        patience = config["training"].get("patience", 5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")