import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

class FinetuneModel(nn.Module):
    """基于简单CNN的微调模型（避免内存问题）"""
    def __init__(self, num_classes=10):
        super(FinetuneModel, self).__init__()
        # 使用简单的CNN结构，避免ResNet的内存问题
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(48 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class SimpleCNN(nn.Module):
    """简单的CNN模型，用于从头训练对比"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def train_model(model, train_loader, device, num_epochs=10, lr=1e-3, weight_decay=1e-4):
    """训练模型"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for imgs, labels in pbar:
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
            
            pbar.set_postfix({
                'Loss': f'{total_loss/(total/imgs.size(0)):.4f}',
                'Acc': f'{correct/total:.4f}'
            })
    
    return model

def evaluate_model(model, data_loader, device):
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
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(images.transpose(0, 3, 1, 2)),  # (N, H, W, C) -> (N, C, H, W)
        torch.from_numpy(labels)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def main():
    """微调模型训练示例 - 实现真正的迁移学习"""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from scripts.data_loader import DataLoader
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    data_loader = DataLoader()
    dataset = data_loader.get_domain_dataset(source='mnist', target='mnistm')
    
    src_train_imgs, src_train_labels = dataset['source']['train']
    tgt_train_imgs, tgt_train_labels = dataset['target']['train']
    tgt_test_imgs, tgt_test_labels = dataset['target']['test']
    
    # 限制数据量以加快训练和避免内存问题
    src_train_imgs = src_train_imgs[:5000]
    src_train_labels = src_train_labels[:5000]
    tgt_train_imgs = tgt_train_imgs[:5000]
    tgt_train_labels = tgt_train_labels[:5000]
    tgt_test_imgs = tgt_test_imgs[:1000]
    tgt_test_labels = tgt_test_labels[:1000]
    
    print(f"\n数据量:")
    print(f"  源域训练: {len(src_train_labels)}")
    print(f"  目标域训练: {len(tgt_train_labels)}")
    print(f"  目标域测试: {len(tgt_test_labels)}")
    
    # 创建数据加载器（使用较小的batch_size避免内存问题）
    batch_size = 32
    src_train_loader = get_loader(src_train_imgs, src_train_labels, batch_size=batch_size)
    tgt_train_loader = get_loader(tgt_train_imgs, tgt_train_labels, batch_size=batch_size)
    tgt_test_loader = get_loader(tgt_test_imgs, tgt_test_labels, batch_size=batch_size, shuffle=False)
    
    # 初始化模型（使用简单CNN，避免ResNet的内存问题）
    model = FinetuneModel(num_classes=10)
    
    # 阶段1: 在源域上预训练
    print("\n=== 阶段1: 在源域(MNIST)上预训练 ===")
    model = train_model(model, src_train_loader, device, num_epochs=5, lr=1e-3)
    
    # 评估源域预训练后的模型在目标域上的表现
    src_pretrained_acc = evaluate_model(model, tgt_test_loader, device)
    print(f"源域预训练后目标域准确率: {src_pretrained_acc:.4f}")
    
    # 阶段2: 在目标域上微调
    print("\n=== 阶段2: 在目标域(MNIST-M)上微调 ===")
    model = train_model(model, tgt_train_loader, device, num_epochs=5, lr=1e-4)
    
    # 评估微调后的模型
    finetuned_acc = evaluate_model(model, tgt_test_loader, device)
    print(f"\n微调后目标域测试准确率: {finetuned_acc:.4f}")
    
    # 保存模型
    model_path = os.path.join(os.path.dirname(__file__), 'finetune_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")
    
    # 输出改进效果
    improvement = finetuned_acc - src_pretrained_acc
    print(f"\n微调带来的准确率提升: {improvement:.4f} ({improvement*100:.2f}%)")

if __name__ == '__main__':
    import os
    main()