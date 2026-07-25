import os
import time
import logging
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = config.get('evaluation.results_dir', './results')
        os.makedirs(self.results_dir, exist_ok=True)

    def evaluate_model(self, model, dataloader):
        model.eval()
        all_preds = []
        all_labels = []
        all_logits = []
        total_loss = 0.0
        start_time = time.time()
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                loss, logits = model(input_ids, attention_mask, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        eval_time = time.time() - start_time
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'loss': total_loss / len(dataloader),
            'preds': all_preds,
            'labels': all_labels,
            'logits': all_logits,
            'eval_time': eval_time,
            'num_samples': len(all_labels)
        }

    def compute_forgetting_rate(self, task_accuracies):
        """
        Compute forgetting rate across tasks.
        
        Args:
            task_accuracies: List of dicts, each containing accuracy after each task
                            e.g., [{task1: 0.8}, {task1: 0.75, task2: 0.85}, ...]
        
        Returns:
            forgetting_rate: Dict of forgetting rates per task
        """
        forgetting = {}
        
        for task_idx, task_name in enumerate(task_accuracies[0].keys()):
            max_acc = task_accuracies[task_idx][task_name]
            final_acc = task_accuracies[-1][task_name]
            forgetting[task_name] = max_acc - final_acc
        
        return forgetting

    def compute_memory_usage(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            return {
                'allocated_mb': allocated,
                'reserved_mb': reserved
            }
        return {'allocated_mb': 0, 'reserved_mb': 0}

    def save_results(self, results, filename='evaluation_results.npy'):
        filepath = os.path.join(self.results_dir, filename)
        np.save(filepath, results)
        logger.info(f"Results saved to {filepath}")

    def evaluate_continual_learning(self, model, task_loaders, task_names):
        """
        Evaluate model on all tasks after each task training.
        
        Args:
            model: The trained model
            task_loaders: List of test dataloaders for each task
            task_names: List of task names
        
        Returns:
            task_accuracies: List of dicts containing accuracy on each task after each training step
        """
        task_accuracies = []
        
        for task_idx in range(len(task_names)):
            accuracies = {}
            for eval_task_idx in range(task_idx + 1):
                results = self.evaluate_model(model, task_loaders[eval_task_idx])
                accuracies[task_names[eval_task_idx]] = results['accuracy']
            task_accuracies.append(accuracies)
            logger.info(f"After task {task_idx + 1}, accuracies: {accuracies}")
        
        return task_accuracies