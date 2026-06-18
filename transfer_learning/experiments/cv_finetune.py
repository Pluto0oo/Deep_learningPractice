"""
CV迁移学习微调实验 - MNIST -> MNIST-M
引用 models/finetune.py 中的模型定义
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gzip
import pickle
import os

# 引用 models 目录中的模型定义
from models.finetune import SimpleCNN, FinetuneModel

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
    if images.ndim == 4 and images.shape[3] == 3:
        images = images.transpose(0, 3, 1, 2)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(images),
        torch.from_numpy(labels)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def main():
    """主函数 - 运行CV微调实验"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    print("加载数据...")
    mnist_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist', 'MNIST', 'raw')
    
    with gzip.open(os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz'), 'rb') as f:
        train_imgs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(os.path.join(mnist_dir, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    
    mnistm_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnistm', 'mnistm_data.pkl')
    with open(mnistm_path, 'rb') as f:
        mnistm_data = pickle.load(f)
    
    tgt_train_imgs = mnistm_data['train'].astype(np.float32) / 255.0
    tgt_train_labels = mnistm_data['train_labels']
    tgt_test_imgs = mnistm_data['test'].astype(np.float32) / 255.0
    tgt_test_labels = mnistm_data['test_labels']
    
    train_imgs = train_imgs.astype(np.float32) / 255.0
    train_imgs = np.repeat(train_imgs[:, np.newaxis, :, :], 3, axis=1)
    
    print(f"源域训练数据: {train_imgs.shape}")
    print(f"目标域训练数据: {tgt_train_imgs.shape}")
    print(f"目标域测试数据: {tgt_test_imgs.shape}\n")
    
    source_loader = get_loader(train_imgs[:10000], train_labels[:10000])
    target_train_loader = get_loader(tgt_train_imgs[:10000], tgt_train_labels[:10000])
    tgt_test_loader = get_loader(tgt_test_imgs, tgt_test_labels, shuffle=False)
    
    # 使用 models/finetune.py 中的 SimpleCNN 模型
    model = SimpleCNN(num_classes=10)
    
    model = pretrain_model(model, source_loader, device, num_epochs=10)
    model = finetune_model(model, target_train_loader, device, num_epochs=5)
    
    print("\n在目标域评估...")
    accuracy = evaluate(model, tgt_test_loader, device)
    print(f"\n目标域测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, 'cv_finetune_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    main()