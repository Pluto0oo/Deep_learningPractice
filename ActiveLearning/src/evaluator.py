import torch
import numpy as np
from typing import Dict, Any


def evaluate_model(model, data: Dict[str, np.ndarray], criterion, 
                   device: torch.device) -> Dict[str, float]:
    model.eval()
    
    X = torch.tensor(data["X"], dtype=torch.float32).to(device)
    y = torch.tensor(data["y"], dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y).item()
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / y.size(0)
    
    return {
        "loss": loss,
        "accuracy": accuracy,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    
    return metrics


def predict_model(model, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    return predicted.cpu().numpy()