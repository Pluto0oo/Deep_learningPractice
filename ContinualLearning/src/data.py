import os
import logging
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
from datasets import load_dataset

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_seq_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ContinualDataLoader:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.data_dir = config.get('dataset.data_dir', './data')
        self.batch_size = config.get('dataset.batch_size', 32)
        self.val_split = config.get('dataset.val_split', 0.1)
        
        os.makedirs(self.data_dir, exist_ok=True)

    def load_task(self, task_name, num_classes, max_seq_len=128):
        logger.info(f"Loading dataset: {task_name}")
        
        if task_name == 'imdb':
            dataset = load_dataset('imdb', 'plain_text', split='train')
            texts = dataset['text']
            labels = dataset['label']
        elif task_name == 'ag_news':
            dataset = load_dataset('ag_news', split='train')
            texts = dataset['text']
            labels = dataset['label']
        elif task_name == 'dbpedia_14':
            dataset = load_dataset('dbpedia_14', split='train')
            texts = dataset['text']
            labels = dataset['label']
        else:
            raise ValueError(f"Unknown dataset: {task_name}")
        
        sample_size = self.config.get('dataset.sample_size', None)
        if sample_size is not None and sample_size < len(texts):
            logger.info(f"Sampling {sample_size} examples from {len(texts)}")
            indices = np.random.choice(len(texts), sample_size, replace=False)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        dataset = TextDataset(texts, labels, self.tokenizer, max_seq_len)
        
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader

    def load_test_task(self, task_name, num_classes, max_seq_len=128):
        logger.info(f"Loading test dataset: {task_name}")
        
        if task_name == 'imdb':
            dataset = load_dataset('imdb', 'plain_text', split='test')
            texts = dataset['text']
            labels = dataset['label']
        elif task_name == 'ag_news':
            dataset = load_dataset('ag_news', split='test')
            texts = dataset['text']
            labels = dataset['label']
        elif task_name == 'dbpedia_14':
            dataset = load_dataset('dbpedia_14', split='test')
            texts = dataset['text']
            labels = dataset['label']
        else:
            raise ValueError(f"Unknown dataset: {task_name}")
        
        sample_size = self.config.get('dataset.sample_size', None)
        if sample_size is not None and sample_size < len(texts):
            logger.info(f"Sampling {sample_size} test examples from {len(texts)}")
            indices = np.random.choice(len(texts), sample_size, replace=False)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        dataset = TextDataset(texts, labels, self.tokenizer, max_seq_len)
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        return test_loader


class ExperienceBuffer:
    def __init__(self, buffer_size=1000, sample_strategy='uniform'):
        self.buffer_size = buffer_size
        self.sample_strategy = sample_strategy
        self.buffer = []

    def add(self, batch):
        for i in range(len(batch['input_ids'])):
            if len(self.buffer) >= self.buffer_size:
                self.buffer.pop(0)
            self.buffer.append({
                'input_ids': batch['input_ids'][i].clone(),
                'attention_mask': batch['attention_mask'][i].clone(),
                'labels': batch['labels'][i].clone()
            })

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None
        
        if self.sample_strategy == 'uniform':
            indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        else:
            indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        
        samples = [self.buffer[i] for i in indices]
        
        return {
            'input_ids': torch.stack([s['input_ids'] for s in samples]),
            'attention_mask': torch.stack([s['attention_mask'] for s in samples]),
            'labels': torch.stack([s['labels'] for s in samples])
        }

    def __len__(self):
        return len(self.buffer)