"""
迁移学习评估对比实验
对比三种方法：
1. 基线模型（仅源域训练）
2. 微调模型（源域预训练 + 目标域微调）
3. DANN模型（域对抗神经网络）
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import os

# 导入统一的模型定义
from models.dann import DANN
from models.finetune import SimpleCNN

def get_loader(images, labels, batch_size=64, shuffle=True):
    """创建数据加载器"""
    images_copy = images.copy()
    labels_copy = labels.copy()
    
    if images_copy.ndim == 4 and images_copy.shape[3] == 3:
        images_copy = np.transpose(images_copy, (0, 3, 1, 2))
    
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(images_copy),
        torch.from_numpy(labels_copy)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def load_data():
    """加载MNIST和MNIST-M数据集"""
    import gzip
    import pickle
    
    mnist_dir = 'data/mnist/MNIST/raw/'
    
    with gzip.open(mnist_dir + 'train-images-idx3-ubyte.gz', 'rb') as f:
        src_train_imgs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(mnist_dir + 'train-labels-idx1-ubyte.gz', 'rb') as f:
        src_train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(mnist_dir + 't10k-images-idx3-ubyte.gz', 'rb') as f:
        src_test_imgs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(mnist_dir + 't10k-labels-idx1-ubyte.gz', 'rb') as f:
        src_test_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    
    src_train_imgs = src_train_imgs.astype(np.float32) / 255.0
    src_train_imgs = np.repeat(src_train_imgs[:, np.newaxis, :, :], 3, axis=1)
    src_test_imgs = src_test_imgs.astype(np.float32) / 255.0
    src_test_imgs = np.repeat(src_test_imgs[:, np.newaxis, :, :], 3, axis=1)
    
    with open('data/mnistm/mnistm_data.pkl', 'rb') as f:
        mnistm_data = pickle.load(f)
    
    tgt_train_imgs = mnistm_data['train'].astype(np.float32) / 255.0
    tgt_train_labels = mnistm_data['train_labels']
    tgt_test_imgs = mnistm_data['test'].astype(np.float32) / 255.0
    tgt_test_labels = mnistm_data['test_labels']
    
    src_train_imgs = src_train_imgs[:10000]
    src_train_labels = src_train_labels[:10000]
    tgt_train_imgs = tgt_train_imgs[:10000]
    tgt_train_labels = tgt_train_labels[:10000]
    tgt_test_imgs = tgt_test_imgs[:2000]
    tgt_test_labels = tgt_test_labels[:2000]
    
    return {
        'src_train': (src_train_imgs, src_train_labels),
        'src_test': (src_test_imgs, src_test_labels),
        'tgt_train': (tgt_train_imgs, tgt_train_labels),
        'tgt_test': (tgt_test_imgs, tgt_test_labels)
    }

def train_dann(model, src_loader, tgt_loader, device, epochs=10):
    """训练DANN模型"""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    model.train()
    
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_loader)
    
    num_batches = min(len(src_loader), len(tgt_loader))
    
    for epoch in range(epochs):
        total_loss = 0.0
        class_correct = 0
        total = 0
        
        p = epoch / epochs
        alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1
        
        for i in range(num_batches):
            try:
                src_imgs, src_labels = next(src_iter)
            except:
                src_iter = iter(src_loader)
                src_imgs, src_labels = next(src_iter)
            
            try:
                tgt_imgs, _ = next(tgt_iter)
            except:
                tgt_iter = iter(tgt_loader)
                tgt_imgs, _ = next(tgt_iter)
            
            batch_size = src_imgs.size(0)
            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
            tgt_imgs = tgt_imgs.to(device)
            
            imgs = torch.cat([src_imgs, tgt_imgs], 0)
            domain_labels = torch.cat([
                torch.zeros(batch_size, dtype=torch.long),
                torch.ones(batch_size, dtype=torch.long)
            ]).to(device)
            
            optimizer.zero_grad()
            
            class_out, domain_out = model(imgs, alpha)
            
            loss = criterion(class_out[:batch_size], src_labels) + 0.1 * criterion(domain_out, domain_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = torch.max(class_out[:batch_size], 1)
            class_correct += (pred == src_labels).sum().item()
            total += batch_size
        
        acc = class_correct / total
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"DANN Epoch {epoch+1}/{epochs} | Loss: {total_loss/num_batches:.4f} | Acc: {acc:.4f}")
    
    return model

def train_baseline(model, train_loader, device, epochs=10):
    """训练基线模型"""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in train_loader:
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
        
        acc = correct / total
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Baseline Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f}")
    
    return model

def train_finetune(model, src_loader, tgt_loader, device, pretrain_epochs=10, finetune_epochs=5):
    """训练微调模型"""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    
    print("预训练阶段...")
    model.train()
    for epoch in range(pretrain_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in src_loader:
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
        
        acc = correct / total
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Pretrain Epoch {epoch+1}/{pretrain_epochs} | Loss: {total_loss/len(src_loader):.4f} | Acc: {acc:.4f}")
    
    print("\n微调阶段...")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(finetune_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in tgt_loader:
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
        
        acc = correct / total
        print(f"Finetune Epoch {epoch+1}/{finetune_epochs} | Loss: {total_loss/len(tgt_loader):.4f} | Acc: {acc:.4f}")
    
    return model

def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if isinstance(model, DANN):
                outputs, _ = model(imgs)
            else:
                outputs = model(imgs)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

def main():
    """主函数 - 运行完整评估对比实验"""
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f"工作目录: {os.getcwd()}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    print("=" * 60)
    print("迁移学习评估对比实验")
    print("实验任务: MNIST -> MNIST-M")
    print("=" * 60)
    
    start_time = time.time()
    
    print("\n加载数据...")
    data = load_data()
    
    src_train_imgs, src_train_labels = data['src_train']
    tgt_train_imgs, tgt_train_labels = data['tgt_train']
    tgt_test_imgs, tgt_test_labels = data['tgt_test']
    
    print(f"源域训练数据: {src_train_imgs.shape}")
    print(f"目标域训练数据: {tgt_train_imgs.shape}")
    print(f"目标域测试数据: {tgt_test_imgs.shape}\n")
    
    src_loader = get_loader(src_train_imgs[:10000], src_train_labels[:10000])
    tgt_train_loader = get_loader(tgt_train_imgs[:10000], tgt_train_labels[:10000])
    tgt_test_loader = get_loader(tgt_test_imgs, tgt_test_labels, shuffle=False)
    
    results = {}
    
    print("\n" + "=" * 60)
    print("实验1: 基线模型 (仅源域训练)")
    print("=" * 60)
    baseline_start = time.time()
    baseline_model = SimpleCNN(num_classes=10)
    baseline_model = train_baseline(baseline_model, src_loader, device, epochs=10)
    baseline_acc = evaluate(baseline_model, tgt_test_loader, device)
    baseline_time = time.time() - baseline_start
    results['baseline'] = {
        'accuracy': baseline_acc,
        'time': baseline_time
    }
    print(f"基线模型目标域准确率: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"耗时: {baseline_time:.2f}秒\n")
    
    print("\n" + "=" * 60)
    print("实验2: 微调模型 (源域预训练 + 目标域微调)")
    print("=" * 60)
    finetune_start = time.time()
    finetune_model = SimpleCNN(num_classes=10)
    finetune_model = train_finetune(finetune_model, src_loader, tgt_train_loader, device, pretrain_epochs=10, finetune_epochs=5)
    finetune_acc = evaluate(finetune_model, tgt_test_loader, device)
    finetune_time = time.time() - finetune_start
    results['finetune'] = {
        'accuracy': finetune_acc,
        'time': finetune_time
    }
    print(f"微调模型目标域准确率: {finetune_acc:.4f} ({finetune_acc*100:.2f}%)")
    print(f"耗时: {finetune_time:.2f}秒\n")
    
    print("\n" + "=" * 60)
    print("实验3: DANN模型 (域对抗神经网络)")
    print("=" * 60)
    dann_start = time.time()
    dann_model = DANN(num_classes=10)
    dann_model = train_dann(dann_model, src_loader, tgt_train_loader, device, epochs=10)
    dann_acc = evaluate(dann_model, tgt_test_loader, device)
    dann_time = time.time() - dann_start
    results['dann'] = {
        'accuracy': dann_acc,
        'time': dann_time
    }
    print(f"DANN模型目标域准确率: {dann_acc:.4f} ({dann_acc*100:.2f}%)")
    print(f"耗时: {dann_time:.2f}秒\n")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print(f"基线模型准确率:   {baseline_acc:.4f} ({baseline_acc*100:.2f}%) | 耗时: {baseline_time:.2f}秒")
    print(f"微调模型准确率:   {finetune_acc:.4f} ({finetune_acc*100:.2f}%) | 耗时: {finetune_time:.2f}秒")
    print(f"DANN模型准确率:   {dann_acc:.4f} ({dann_acc*100:.2f}%) | 耗时: {dann_time:.2f}秒")
    print(f"\n总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    
    os.makedirs('experiments/mnist_to_mnistm', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    with open('experiments/mnist_to_mnistm/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    torch.save(baseline_model.state_dict(), 'results/baseline_final.pth')
    torch.save(finetune_model.state_dict(), 'results/finetune_final.pth')
    torch.save(dann_model.state_dict(), 'results/dann_final.pth')
    
    print("\n结果已保存到:")
    print("- 实验结果: experiments/mnist_to_mnistm/comparison_results.json")
    print("- 模型文件: results/")

def run_dann_experiment():
    """单独运行DANN实验"""
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    print("=" * 60)
    print("DANN模型实验 (域对抗神经网络)")
    print("实验任务: MNIST -> MNIST-M")
    print("=" * 60)
    
    print("\n加载数据...")
    data = load_data()
    
    src_train_imgs, src_train_labels = data['src_train']
    tgt_train_imgs, tgt_train_labels = data['tgt_train']
    tgt_test_imgs, tgt_test_labels = data['tgt_test']
    
    print(f"源域训练数据: {src_train_imgs.shape}")
    print(f"目标域训练数据: {tgt_train_imgs.shape}")
    print(f"目标域测试数据: {tgt_test_imgs.shape}\n")
    
    src_loader = get_loader(src_train_imgs[:10000], src_train_labels[:10000])
    tgt_train_loader = get_loader(tgt_train_imgs[:10000], tgt_train_labels[:10000])
    tgt_test_loader = get_loader(tgt_test_imgs, tgt_test_labels, shuffle=False)
    
    dann_start = time.time()
    dann_model = DANN(num_classes=10)
    dann_model = train_dann(dann_model, src_loader, tgt_train_loader, device, epochs=10)
    dann_acc = evaluate(dann_model, tgt_test_loader, device)
    dann_time = time.time() - dann_start
    
    print(f"\nDANN模型目标域准确率: {dann_acc:.4f} ({dann_acc*100:.2f}%)")
    print(f"耗时: {dann_time:.2f}秒\n")
    
    os.makedirs('results', exist_ok=True)
    torch.save(dann_model.state_dict(), 'results/dann_final.pth')
    print("模型已保存到: results/dann_final.pth")

if __name__ == '__main__':
    main()
