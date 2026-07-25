import os
import sys
import time
import logging
import argparse
import torch
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data import ContinualDataLoader
from src.models import LWFModel
from src.trainer import LWFTrainer
from src.evaluator import Evaluator
from src.visualization import Visualizer

logger = logging.getLogger(__name__)


def main(config_path):
    config = Config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.get('model.name', 'distilbert-base-uncased'))
    
    data_loader = ContinualDataLoader(config, tokenizer)
    
    task1_name = config.get('dataset.task1.name', 'imdb')
    task1_classes = config.get('dataset.task1.num_classes', 2)
    task1_max_seq_len = config.get('dataset.task1.max_seq_len', 64)
    
    task2_name = config.get('dataset.task2.name', 'ag_news')
    task2_classes = config.get('dataset.task2.num_classes', 4)
    task2_max_seq_len = config.get('dataset.task2.max_seq_len', 64)
    
    logger.info(f"Loading task 1: {task1_name}")
    train_loader1, val_loader1 = data_loader.load_task(task1_name, task1_classes, task1_max_seq_len)
    
    logger.info(f"Loading task 2: {task2_name}")
    train_loader2, val_loader2 = data_loader.load_task(task2_name, task2_classes, task2_max_seq_len)
    
    logger.info(f"Loading test sets")
    test_loader1 = data_loader.load_test_task(task1_name, task1_classes, task1_max_seq_len)
    test_loader2 = data_loader.load_test_task(task2_name, task2_classes, task2_max_seq_len)
    
    model = LWFModel(
        model_name=config.get('model.name', 'distilbert-base-uncased'),
        num_labels=config.get('model.num_labels', 6),
        dropout_rate=config.get('model.dropout_rate', 0.1)
    )
    
    trainer = LWFTrainer(model, config, device)
    evaluator = Evaluator(config, device)
    
    start_time = time.time()
    
    logger.info("=" * 50)
    logger.info("Training on Task 1")
    logger.info("=" * 50)
    
    train_history1, val_history1 = trainer.train(train_loader1, val_loader1)
    
    logger.info("=" * 50)
    logger.info("Evaluating after Task 1")
    logger.info("=" * 50)
    
    results_after_task1 = {}
    results_after_task1[task1_name] = evaluator.evaluate_model(model, test_loader1)
    logger.info(f"Task 1 accuracy: {results_after_task1[task1_name]['accuracy']:.4f}")
    
    old_model = LWFModel(
        model_name=config.get('model.name', 'distilbert-base-uncased'),
        num_labels=config.get('model.num_labels', 6),
        dropout_rate=config.get('model.dropout_rate', 0.1)
    )
    old_model.load_state_dict(model.state_dict())
    model.set_old_model(old_model)
    
    logger.info("=" * 50)
    logger.info("Training on Task 2 with LWF")
    logger.info("=" * 50)
    
    train_history2, val_history2 = trainer.train(train_loader2, val_loader2)
    
    logger.info("=" * 50)
    logger.info("Evaluating after Task 2")
    logger.info("=" * 50)
    
    results_after_task2 = {}
    results_after_task2[task1_name] = evaluator.evaluate_model(model, test_loader1)
    results_after_task2[task2_name] = evaluator.evaluate_model(model, test_loader2)
    
    logger.info(f"Task 1 accuracy: {results_after_task2[task1_name]['accuracy']:.4f}")
    logger.info(f"Task 2 accuracy: {results_after_task2[task2_name]['accuracy']:.4f}")
    
    total_time = time.time() - start_time
    
    task_accuracies = [
        {task1_name: results_after_task1[task1_name]['accuracy']},
        {task1_name: results_after_task2[task1_name]['accuracy'], task2_name: results_after_task2[task2_name]['accuracy']}
    ]
    
    forgetting_rates = evaluator.compute_forgetting_rate(task_accuracies)
    
    logger.info(f"\n{'=' * 50}")
    logger.info("FINAL RESULTS - LWF")
    logger.info(f"{'=' * 50}")
    logger.info(f"Task 1 final accuracy: {results_after_task2[task1_name]['accuracy']:.4f}")
    logger.info(f"Task 2 final accuracy: {results_after_task2[task2_name]['accuracy']:.4f}")
    logger.info(f"Forgetting rate (Task 1): {forgetting_rates[task1_name]:.4f}")
    logger.info(f"Total training time: {total_time:.2f}s")
    
    memory_usage = evaluator.compute_memory_usage()
    logger.info(f"Memory usage - Allocated: {memory_usage['allocated_mb']:.2f}MB, Reserved: {memory_usage['reserved_mb']:.2f}MB")
    
    visualizer = Visualizer(config)
    visualizer.plot_training_curves(
        [{'train': train_history1, 'val': val_history1}, {'train': train_history2, 'val': val_history2}],
        ['Task 1', 'Task 2'],
        filename='lwf_curves.png'
    )
    
    results = {
        'method': 'lwf',
        'task_accuracies': task_accuracies,
        'forgetting_rates': forgetting_rates,
        'total_time': total_time,
        'memory_usage': memory_usage,
        'train_history': {'task1': train_history1, 'task2': train_history2},
        'final_accuracies': {
            task1_name: results_after_task2[task1_name]['accuracy'],
            task2_name: results_after_task2[task2_name]['accuracy']
        }
    }
    
    evaluator.save_results(results, filename='lwf_results.npy')
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LWF Continual Learning Experiment')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)