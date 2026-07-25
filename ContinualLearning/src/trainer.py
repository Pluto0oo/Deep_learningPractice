import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.num_epochs = int(config.get('training.num_epochs', 3))
        self.learning_rate = float(config.get('model.learning_rate', 2e-5))
        self.weight_decay = float(config.get('model.weight_decay', 1e-4))
        self.warmup_ratio = float(config.get('training.warmup_ratio', 0.1))
        self.gradient_accumulation_steps = int(config.get('training.gradient_accumulation_steps', 1))
        self.max_grad_norm = float(config.get('training.max_grad_norm', 1.0))
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=float(config.get('model.adam_epsilon', 1e-8))
        )
        
        self.train_history = []
        self.val_history = []

    def _setup_scheduler(self, total_steps):
        warmup_steps = int(total_steps * self.warmup_ratio)
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    def train_epoch(self, dataloader, scheduler=None):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            loss, logits = self.model(input_ids, attention_mask, labels)
            
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            loss.backward()
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if scheduler:
                    scheduler.step()
                self.model.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                loss, logits = self.model(input_ids, attention_mask, labels)
                
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy

    def train(self, train_loader, val_loader):
        total_steps = len(train_loader) * self.num_epochs
        scheduler = self._setup_scheduler(total_steps)
        
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(train_loader, scheduler)
            val_loss, val_acc = self.evaluate(val_loader)
            
            epoch_time = time.time() - start_time
            
            self.train_history.append({'loss': train_loss, 'accuracy': train_acc})
            self.val_history.append({'loss': val_loss, 'accuracy': val_acc})
            
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            logger.info(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            logger.info(f"  Time: {epoch_time:.2f}s")
        
        return self.train_history, self.val_history

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")


class FineTuningTrainer(Trainer):
    def __init__(self, model, config, device=None):
        super().__init__(model, config, device)


class EWCTrainer(Trainer):
    def __init__(self, model, config, device=None):
        super().__init__(model, config, device)
        self.lambda_ewc = config.get('continual_methods.ewc.lambda', 1000.0)

    def compute_fisher_information(self, dataloader):
        return self.model.compute_fisher(dataloader, self.device)

    def train_epoch(self, dataloader, scheduler=None):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            loss, logits = self.model(input_ids, attention_mask, labels)
            
            if self.model.fisher_information is not None and self.model.old_params is not None:
                ewc_loss = 0.0
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in self.model.fisher_information:
                        ewc_loss += 0.5 * self.lambda_ewc * \
                            torch.sum(self.model.fisher_information[name] * 
                                      (param - self.model.old_params[name]) ** 2)
                loss += ewc_loss
            
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            loss.backward()
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if scheduler:
                    scheduler.step()
                self.model.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy


class LWFTrainer(Trainer):
    def __init__(self, model, config, device=None):
        super().__init__(model, config, device)
        self.temperature = config.get('continual_methods.lwf.temperature', 2.0)
        self.alpha = config.get('continual_methods.lwf.alpha', 0.5)

    def train_epoch(self, dataloader, scheduler=None):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            loss, logits = self.model(input_ids, attention_mask, labels)
            
            if self.model.old_model is not None:
                with torch.no_grad():
                    _, old_logits = self.model.old_model(input_ids, attention_mask)
                
                soft_targets = F.softmax(old_logits / self.temperature, dim=1)
                soft_preds = F.log_softmax(logits / self.temperature, dim=1)
                distillation_loss = F.kl_div(soft_preds, soft_targets, reduction='batchmean')
                
                loss = (1 - self.alpha) * loss + self.alpha * (self.temperature ** 2) * distillation_loss
            
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            loss.backward()
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if scheduler:
                    scheduler.step()
                self.model.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy


class DERTrainer(Trainer):
    def __init__(self, model, config, device=None):
        super().__init__(model, config, device)
        self.alpha = config.get('continual_methods.der.alpha', 0.5)
        self.beta = config.get('continual_methods.der.beta', 0.5)

    def train_epoch(self, dataloader, scheduler=None):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            loss, logits = self.model(input_ids, attention_mask, labels)
            
            total_loss_ce = loss.item()
            
            if self.model.buffer is not None and len(self.model.buffer) > 0:
                buffer_batch = self.model.buffer.sample(len(labels))
                if buffer_batch is not None:
                    buf_input_ids = buffer_batch['input_ids'].to(self.device)
                    buf_attention_mask = buffer_batch['attention_mask'].to(self.device)
                    buf_labels = buffer_batch['labels'].to(self.device)
                    
                    buf_loss, buf_logits = self.model(buf_input_ids, buf_attention_mask, buf_labels)
                    loss += self.alpha * buf_loss
                    
                    if self.model.old_model is not None:
                        with torch.no_grad():
                            _, old_buf_logits = self.model.old_model(buf_input_ids, buf_attention_mask)
                        
                        soft_targets = F.softmax(old_buf_logits / 2.0, dim=1)
                        soft_preds = F.log_softmax(buf_logits / 2.0, dim=1)
                        distillation_loss = F.kl_div(soft_preds, soft_targets, reduction='batchmean')
                        loss += self.beta * (2.0 ** 2) * distillation_loss
            
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            loss.backward()
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if scheduler:
                    scheduler.step()
                self.model.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy