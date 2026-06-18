import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Function
from tqdm import tqdm

class GradientReversalFunction(Function):
    """梯度反转函数"""
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversal(nn.Module):
    """梯度反转层 - 在反向传播时反转梯度符号"""
    def __init__(self, alpha=1.0):
        super(GradientReversal, self).__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class FeatureExtractor(nn.Module):
    """特征提取器 - 用于提取图像特征"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.conv_layers(x)

class LabelClassifier(nn.Module):
    """标签分类器 - 用于分类任务"""
    def __init__(self, num_classes=10):
        super(LabelClassifier, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(48 * 7 * 7, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

class DomainClassifier(nn.Module):
    """域分类器 - 用于区分源域和目标域"""
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(48 * 7 * 7, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2)
        )
    
    def forward(self, x, alpha=1.0):
        x = x.view(x.size(0), -1)
        x = GradientReversalFunction.apply(x, alpha)
        return self.fc_layers(x)

class DANN(nn.Module):
    """Domain-Adversarial Neural Network"""
    def __init__(self, num_classes=10):
        super(DANN, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.label_classifier = LabelClassifier(num_classes)
        self.domain_classifier = DomainClassifier()
    
    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        class_output = self.label_classifier(features)
        domain_output = self.domain_classifier(features, alpha)
        return class_output, domain_output

def train_dann(model, source_loader, target_loader, device, num_epochs=20, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    
    model.to(device)
    model.train()
    
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        class_correct = 0
        domain_correct = 0
        total = 0
        
        num_batches = min(len(source_loader), len(target_loader))
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        for i in range(num_batches):
            try:
                src_imgs, src_labels = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                src_imgs, src_labels = next(source_iter)
            
            try:
                tgt_imgs, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_imgs, _ = next(target_iter)
            
            batch_size = src_imgs.size(0)
            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
            tgt_imgs = tgt_imgs.to(device)
            
            imgs = torch.cat([src_imgs, tgt_imgs], dim=0)
            
            domain_labels = torch.cat([
                torch.zeros(batch_size, dtype=torch.long),
                torch.ones(batch_size, dtype=torch.long)
            ]).to(device)
            
            p = float(i + epoch * num_batches) / (num_epochs * num_batches)
            alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1
            
            optimizer.zero_grad()
            
            class_output, domain_output = model(imgs, alpha)
            
            class_loss = criterion_class(class_output[:batch_size], src_labels)
            domain_loss = criterion_domain(domain_output, domain_labels)
            
            loss = class_loss + domain_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, class_pred = torch.max(class_output[:batch_size], 1)
            class_correct += (class_pred == src_labels).sum().item()
            
            _, domain_pred = torch.max(domain_output, 1)
            domain_correct += (domain_pred == domain_labels).sum().item()
            
            total += batch_size
        
        avg_loss = total_loss / num_batches
        class_acc = class_correct / total
        domain_acc = domain_correct / (2 * total)
        
        print(f"Loss: {avg_loss:.4f} | Class Acc: {class_acc:.4f} | Domain Acc: {domain_acc:.4f} | Alpha: {alpha:.4f}")
    
    return model

def evaluate_dann(model, data_loader, device, task='class'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            if task == 'class':
                class_output, _ = model(imgs)
                _, pred = torch.max(class_output, 1)
            elif task == 'domain':
                _, domain_output = model(imgs)
                _, pred = torch.max(domain_output, 1)
            else:
                raise ValueError(f"Unknown task: {task}")
            
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

def get_loader(images, labels, batch_size=64, shuffle=True):
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(images.transpose(0, 3, 1, 2)),
        torch.from_numpy(labels)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

