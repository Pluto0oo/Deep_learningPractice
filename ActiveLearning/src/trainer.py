import os
import csv
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from .model import build_model, get_loss_function, get_optimizer, get_scheduler
from .evaluator import evaluate_model


def train_model(config: Dict[str, Any], train_data: Dict[str, np.ndarray], 
                device: torch.device, logger, exp_dirs: Dict[str, str]) -> Dict[str, Any]:
    model = build_model(config).to(device)
    criterion = get_loss_function(config)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    
    training_config = config["training"]
    epochs = training_config.get("epochs", 100)
    batch_size = training_config.get("batch_size", 32)
    save_best = training_config.get("save_best", True)
    
    X_train = torch.tensor(train_data["X"], dtype=torch.float32).to(device)
    y_train = torch.tensor(train_data["y"], dtype=torch.long).to(device)
    
    if "X_val" in train_data:
        X_val = torch.tensor(train_data["X_val"], dtype=torch.float32).to(device)
        y_val = torch.tensor(train_data["y_val"], dtype=torch.long).to(device)
        val_data = {"X": X_val, "y": y_val}
    else:
        val_data = None
    
    best_val_acc = 0.0
    best_epoch = 0
    epoch_metrics = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        permutation = torch.randperm(X_train.size()[0])
        num_batches = len(permutation) // batch_size
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for i in tqdm(range(num_batches), desc=f"Epoch {epoch}/{epochs}"):
            indices = permutation[i * batch_size : (i + 1) * batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        train_loss = total_loss / num_batches
        train_acc = correct / total
        
        epoch_metric = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
        }
        
        if val_data is not None:
            val_metrics = evaluate_model(model, val_data, criterion, device)
            epoch_metric.update(val_metrics)
            
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_epoch = epoch
                if save_best:
                    checkpoint_path = os.path.join(exp_dirs["checkpoints"], "best_model.pt")
                    model.save_checkpoint(checkpoint_path, epoch, val_metrics)
                    logger.info(f"Best model saved at epoch {epoch} with val_acc: {best_val_acc:.4f}")
            
            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["accuracy"])
        else:
            if save_best and epoch == 1:
                checkpoint_path = os.path.join(exp_dirs["checkpoints"], "best_model.pt")
                model.save_checkpoint(checkpoint_path, epoch, epoch_metric)
        
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        
        epoch_metrics.append(epoch_metric)
        
        log_msg = f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}"
        if val_data is not None:
            log_msg += f", val_loss: {val_metrics['loss']:.4f}, val_acc: {val_metrics['accuracy']:.4f}"
        logger.info(log_msg)
    
    checkpoint_path = os.path.join(exp_dirs["checkpoints"], "final_model.pt")
    model.save_checkpoint(checkpoint_path, epochs)
    
    save_epoch_metrics(epoch_metrics, os.path.join(exp_dirs["base"], "metrics.csv"))
    
    final_metrics = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        **epoch_metrics[-1],
    }
    
    return final_metrics


def save_epoch_metrics(epoch_metrics: List[Dict[str, float]], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_metrics[0].keys())
        writer.writeheader()
        writer.writerows(epoch_metrics)