import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TextClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=6, dropout_rate=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.num_labels = num_labels
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        if hasattr(self.encoder, 'encoder'):
            for param in self.encoder.encoder.layer[-2:].parameters():
                param.requires_grad = True
        elif hasattr(self.encoder, 'transformer'):
            for param in self.encoder.transformer.layer[-2:].parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return loss, logits


class EWCModel(TextClassifier):
    def __init__(self, model_name='bert-base-uncased', num_labels=6, dropout_rate=0.1):
        super().__init__(model_name, num_labels, dropout_rate)
        self.fisher_information = None
        self.old_params = None

    def compute_fisher(self, dataloader, device):
        self.eval()
        fisher = {}
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)
        
        total_samples = 0
        for batch in dataloader:
            if total_samples >= 100:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            self.zero_grad()
            loss, _ = self.forward(input_ids, attention_mask, labels)
            loss.backward()
            
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
            
            total_samples += labels.size(0)
        
        for name in fisher:
            fisher[name] /= total_samples
        
        self.fisher_information = fisher
        self.old_params = {name: param.data.clone() for name, param in self.named_parameters() if param.requires_grad}
        
        self.train()
        return fisher


class LWFModel(TextClassifier):
    def __init__(self, model_name='bert-base-uncased', num_labels=6, dropout_rate=0.1):
        super().__init__(model_name, num_labels, dropout_rate)
        self.old_model = None

    def set_old_model(self, old_model):
        self.old_model = old_model
        for param in self.old_model.parameters():
            param.requires_grad = False
        self.old_model.eval()


class DERModel(TextClassifier):
    def __init__(self, model_name='bert-base-uncased', num_labels=6, dropout_rate=0.1):
        super().__init__(model_name, num_labels, dropout_rate)
        self.buffer = None
        self.old_model = None

    def set_buffer_and_old_model(self, buffer, old_model):
        self.buffer = buffer
        self.old_model = old_model
        if self.old_model:
            for param in self.old_model.parameters():
                param.requires_grad = False
            self.old_model.eval()