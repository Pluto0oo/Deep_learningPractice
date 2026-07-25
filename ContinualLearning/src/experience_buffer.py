import random
import torch
from torch.utils.data import DataLoader, TensorDataset


class ExperienceBuffer:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)
    
    def add_batch(self, batch):
        for i in range(batch['input_ids'].size(0)):
            if len(self.buffer) >= self.max_size:
                self.buffer.pop(0)
            self.buffer.append({
                'input_ids': batch['input_ids'][i].clone(),
                'attention_mask': batch['attention_mask'][i].clone(),
                'labels': batch['labels'][i].clone()
            })
    
    def add_sample(self, input_ids, attention_mask, labels):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append({
            'input_ids': input_ids.clone(),
            'attention_mask': attention_mask.clone(),
            'labels': labels.clone()
        })
    
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None
        
        num_samples = min(batch_size, len(self.buffer))
        samples = random.sample(self.buffer, num_samples)
        
        input_ids = torch.stack([s['input_ids'] for s in samples])
        attention_mask = torch.stack([s['attention_mask'] for s in samples])
        labels = torch.tensor([s['labels'].item() for s in samples])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def get_dataloader(self, batch_size=32, shuffle=True):
        if len(self.buffer) == 0:
            return None
        
        input_ids = torch.stack([s['input_ids'] for s in self.buffer])
        attention_mask = torch.stack([s['attention_mask'] for s in self.buffer])
        labels = torch.tensor([s['labels'].item() for s in self.buffer])
        
        dataset = TensorDataset(input_ids, attention_mask, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def clear(self):
        self.buffer = []