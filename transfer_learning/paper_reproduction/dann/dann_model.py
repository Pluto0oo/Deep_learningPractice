"""
DANN (Domain-Adversarial Neural Network) 论文复现
基于论文: "Domain-Adversarial Training of Neural Networks" (ICML 2016)

核心思想：通过对抗训练学习域不变特征，使特征提取器无法区分源域和目标域

注意：本文件保留论文复现特定的训练逻辑，模型定义从统一位置导入
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 从统一位置导入DANN模型定义
from models.dann import DANN

def train_dann(model, source_loader, target_loader, device, num_epochs=20, lr=1e-3, domain_loss_weight=0.1):
    """训练DANN模型 - 优化版本"""
    import numpy as np
    
    # 使用不同的学习率
    optimizer = optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': lr},
        {'params': model.label_classifier.parameters(), 'lr': lr},
        {'params': model.domain_classifier.parameters(), 'lr': lr * 0.1}  # 域分类器使用更小的学习率
    ])
    
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
            
            # 调整alpha调度，更缓慢地增长
            p = float(i + epoch * num_batches) / (num_epochs * num_batches)
            alpha = 2.0 / (1.0 + np.exp(-5.0 * p)) - 1  # 减小指数系数，让alpha增长更慢
            
            optimizer.zero_grad()
            
            class_output, domain_output = model(imgs, alpha)
            
            class_loss = criterion_class(class_output[:batch_size], src_labels)
            domain_loss = criterion_domain(domain_output, domain_labels)
            
            # 使用域损失权重
            loss = class_loss + domain_loss_weight * domain_loss
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
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | "
              f"Class Acc: {class_acc:.4f} | Domain Acc: {domain_acc:.4f} | Alpha: {alpha:.4f}")
    
    return model

def evaluate_dann(model, data_loader, device, task='class'):
    """评估DANN模型"""
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

def load_data():
    """加载MNIST和MNIST-M数据集"""
    import gzip
    import pickle
    import os
    
    mnist_dir = 'data/mnist/MNIST/raw/'
    
    with gzip.open(mnist_dir + 'train-images-idx3-ubyte.gz', 'rb') as f:
        train_imgs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    
    with gzip.open(mnist_dir + 'train-labels-idx1-ubyte.gz', 'rb') as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    
    with gzip.open(mnist_dir + 't10k-images-idx3-ubyte.gz', 'rb') as f:
        test_imgs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    
    with gzip.open(mnist_dir + 't10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    
    train_imgs = train_imgs.astype(np.float32) / 255.0
    train_imgs = np.repeat(train_imgs[:, np.newaxis, :, :], 3, axis=1)
    test_imgs = test_imgs.astype(np.float32) / 255.0
    test_imgs = np.repeat(test_imgs[:, np.newaxis, :, :], 3, axis=1)
    
    with open('data/mnistm/mnistm_data.pkl', 'rb') as f:
        mnistm_data = pickle.load(f)
    
    tgt_train_imgs = mnistm_data['train'].astype(np.float32) / 255.0
    tgt_train_imgs = np.transpose(tgt_train_imgs, (0, 3, 1, 2))
    tgt_train_labels = mnistm_data['train_labels']
    tgt_test_imgs = mnistm_data['test'].astype(np.float32) / 255.0
    tgt_test_imgs = np.transpose(tgt_test_imgs, (0, 3, 1, 2))
    tgt_test_labels = mnistm_data['test_labels']
    
    return {
        'source_train': (train_imgs, train_labels),
        'source_test': (test_imgs, test_labels),
        'target_train': (tgt_train_imgs, tgt_train_labels),
        'target_test': (tgt_test_imgs, tgt_test_labels)
    }

def main():
    """主函数 - 运行DANN实验"""
    import os
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    print("加载数据...")
    data = load_data()
    
    src_train_imgs, src_train_labels = data['source_train']
    tgt_train_imgs, _ = data['target_train']
    tgt_test_imgs, tgt_test_labels = data['target_test']
    
    print(f"源域训练数据: {src_train_imgs.shape}")
    print(f"目标域训练数据: {tgt_train_imgs.shape}")
    print(f"目标域测试数据: {tgt_test_imgs.shape}\n")
    
    source_loader = get_loader(src_train_imgs[:10000], src_train_labels[:10000])
    target_loader = get_loader(tgt_train_imgs[:10000], np.zeros(10000))
    test_loader = get_loader(tgt_test_imgs, tgt_test_labels, shuffle=False)
    
    model = DANN(num_classes=10)
    
    print("训练DANN模型 (10轮)...")
    model = train_dann(model, source_loader, target_loader, device, num_epochs=10)
    
    print("\n评估模型...")
    accuracy = evaluate_dann(model, test_loader, device)
    print(f"\n目标域测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), 'results/dann_paper_model.pth')
    print("模型已保存到: results/dann_paper_model.pth")

if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    main()
