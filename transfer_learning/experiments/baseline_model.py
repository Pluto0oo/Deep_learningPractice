"""
基线模型 - 从头训练的简单CNN
用于与迁移学习方法对比
"""
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    """简单的CNN模型 - 基线对比方法"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
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

def train_baseline(model, train_loader, device, num_epochs=20, lr=1e-3):
    """训练基线模型"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
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
        
        avg_loss = total_loss / len(train_loader)
        acc = correct / total
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
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
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(images.transpose(0, 3, 1, 2)),
        torch.from_numpy(labels)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def main():
    """主函数 - 运行基线实验"""
    import numpy as np
    import gzip
    import pickle
    import os
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    print("加载数据...")
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
    
    tgt_test_imgs = mnistm_data['test'].astype(np.float32) / 255.0
    tgt_test_imgs = np.transpose(tgt_test_imgs, (0, 3, 1, 2))
    tgt_test_labels = mnistm_data['test_labels']
    
    print(f"训练数据: {train_imgs.shape}")
    print(f"目标域测试数据: {tgt_test_imgs.shape}\n")
    
    train_loader = get_loader(train_imgs[:10000], train_labels[:10000])
    tgt_test_loader = get_loader(tgt_test_imgs, tgt_test_labels, shuffle=False)
    
    model = SimpleCNN(num_classes=10)
    
    print("训练基线模型 (10轮)...")
    model = train_baseline(model, train_loader, device, num_epochs=10)
    
    print("\n在目标域评估...")
    accuracy = evaluate(model, tgt_test_loader, device)
    print(f"\n目标域测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), 'results/baseline_model.pth')
    print("模型已保存到: results/baseline_model.pth")

if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    main()
