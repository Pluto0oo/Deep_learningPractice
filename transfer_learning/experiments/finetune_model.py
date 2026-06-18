"""
微调模型 - 在源域预训练后微调到目标域
用于与DANN方法对比
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class FinetuneCNN(nn.Module):
    """微调CNN模型"""
    def __init__(self, num_classes=10):
        super(FinetuneCNN, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class SimpleFinetuneCNN(nn.Module):
    """简单的微调CNN模型 - 不依赖预训练权重"""
    def __init__(self, num_classes=10):
        super(SimpleFinetuneCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(48 * 7 * 7, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

def pretrain_model(model, source_loader, device, num_epochs=10, lr=1e-3):
    """在源域预训练模型"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    model.train()
    
    print("预训练阶段 (源域)...")
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in source_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(source_loader)
        acc = correct / total
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
    
    return model

def finetune_model(model, target_train_loader, device, num_epochs=5, lr=1e-4):
    """在目标域微调模型"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    print("\n微调阶段 (目标域)...")
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in target_train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(target_train_loader)
        acc = correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
    
    return model

def evaluate(model, data_loader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

def get_loader(images, labels, batch_size=64, shuffle=True):
    """创建数据加载器"""
    # 检查数据维度，如果是 (N, H, W, C) 格式则转置为 (N, C, H, W)
    if images.ndim == 4 and images.shape[3] == 3:  # (N, H, W, C)
        images = images.transpose(0, 3, 1, 2)
    # 如果已经是 (N, C, H, W) 格式则保持不变
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(images),
        torch.from_numpy(labels)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def main():
    """主函数 - 运行微调实验"""
    import numpy as np
    import gzip
    import pickle
    import os
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    print("加载数据...")
    mnist_dir = 'transfer_learning/data/mnist/MNIST/raw/'

    with gzip.open(mnist_dir + 'train-images-idx3-ubyte.gz', 'rb') as f:
        train_imgs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(mnist_dir + 'train-labels-idx1-ubyte.gz', 'rb') as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    with open('transfer_learning/data/mnistm/mnistm_data.pkl', 'rb') as f:
        mnistm_data = pickle.load(f)
    
    tgt_train_imgs = mnistm_data['train'].astype(np.float32) / 255.0
    tgt_train_imgs = np.transpose(tgt_train_imgs, (0, 3, 1, 2))
    tgt_train_labels = mnistm_data['train_labels']
    tgt_test_imgs = mnistm_data['test'].astype(np.float32) / 255.0
    tgt_test_imgs = np.transpose(tgt_test_imgs, (0, 3, 1, 2))
    tgt_test_labels = mnistm_data['test_labels']
    
    train_imgs = train_imgs.astype(np.float32) / 255.0
    train_imgs = np.repeat(train_imgs[:, np.newaxis, :, :], 3, axis=1)
    
    print(f"源域训练数据: {train_imgs.shape}")
    print(f"目标域训练数据: {tgt_train_imgs.shape}")
    print(f"目标域测试数据: {tgt_test_imgs.shape}\n")
    
    source_loader = get_loader(train_imgs[:10000], train_labels[:10000])
    target_train_loader = get_loader(tgt_train_imgs[:10000], tgt_train_labels[:10000])
    tgt_test_loader = get_loader(tgt_test_imgs, tgt_test_labels, shuffle=False)
    
    model = SimpleFinetuneCNN(num_classes=10)
    
    model = pretrain_model(model, source_loader, device, num_epochs=10)
    model = finetune_model(model, target_train_loader, device, num_epochs=5)
    
    print("\n在目标域评估...")
    accuracy = evaluate(model, tgt_test_loader, device)
    print(f"\n目标域测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), 'results/finetune_model.pth')
    print("模型已保存到: results/finetune_model.pth")

if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    main()
