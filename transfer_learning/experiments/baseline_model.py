"""
基线模型 - 从头训练的简单CNN
用于与迁移学习方法对比
已集成结果管理器，支持自动保存实验结果
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gzip
import pickle
import os

# 导入结果管理器
try:
    from scripts.result_manager import create_experiment
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from scripts.result_manager import create_experiment

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

def train_baseline(model, train_loader, device, manager, num_epochs=20, lr=1e-3):
    """训练基线模型，集成结果管理器"""
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
        
        # 记录训练日志到结果管理器
        manager.log_training_epoch(
            epoch=epoch + 1,
            loss=avg_loss,
            accuracy=acc,
            learning_rate=lr
        )
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
    
    return model

def evaluate(model, data_loader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return {
        'accuracy': correct / total,
        'correct': correct,
        'total': total,
        'predictions': all_preds,
        'labels': all_labels
    }

def get_loader(images, labels, batch_size=64, shuffle=True):
    """创建数据加载器"""
    if images.ndim == 4 and images.shape[1] != 3:
        images = images.transpose(0, 3, 1, 2)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(images),
        torch.from_numpy(labels)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def main():
    """主函数 - 运行基线实验，自动保存结果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 创建实验结果管理器
    manager = create_experiment('baseline_cnn', {
        'model': 'SimpleCNN',
        'task': 'MNIST -> MNIST-M',
        'dataset': {
            'source': 'MNIST',
            'target': 'MNIST-M',
            'train_samples': 10000,
            'test_samples': 10000
        },
        'hyperparameters': {
            'batch_size': 64,
            'num_epochs': 10,
            'learning_rate': 1e-3,
            'optimizer': 'Adam',
            'loss_function': 'CrossEntropyLoss'
        }
    })
    
    print(f"实验目录: {manager.get_experiment_dir()}\n")
    
    print("加载数据...")
    mnist_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist', 'MNIST', 'raw')
    mnistm_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnistm', 'mnistm_data.pkl')
    
    with gzip.open(os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz'), 'rb') as f:
        train_imgs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(os.path.join(mnist_dir, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(os.path.join(mnist_dir, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
        test_imgs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(os.path.join(mnist_dir, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
        test_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    
    train_imgs = train_imgs.astype(np.float32) / 255.0
    train_imgs = np.repeat(train_imgs[:, np.newaxis, :, :], 3, axis=1)
    test_imgs = test_imgs.astype(np.float32) / 255.0
    test_imgs = np.repeat(test_imgs[:, np.newaxis, :, :], 3, axis=1)
    
    with open(mnistm_path, 'rb') as f:
        mnistm_data = pickle.load(f)
    
    tgt_test_imgs = mnistm_data['test'].astype(np.float32) / 255.0
    tgt_test_labels = mnistm_data['test_labels']
    
    print(f"训练数据: {train_imgs.shape}")
    print(f"目标域测试数据: {tgt_test_imgs.shape}\n")
    
    train_loader = get_loader(train_imgs[:10000], train_labels[:10000])
    tgt_test_loader = get_loader(tgt_test_imgs, tgt_test_labels, shuffle=False)
    
    model = SimpleCNN(num_classes=10)
    
    print("训练基线模型 (10轮)...")
    model = train_baseline(model, train_loader, device, manager, num_epochs=10)
    
    print("\n在源域测试集评估...")
    src_results = evaluate(model, get_loader(test_imgs, test_labels, shuffle=False), device)
    print(f"源域测试准确率: {src_results['accuracy']:.4f} ({src_results['accuracy']*100:.2f}%)")
    
    print("\n在目标域评估...")
    tgt_results = evaluate(model, tgt_test_loader, device)
    print(f"目标域测试准确率: {tgt_results['accuracy']:.4f} ({tgt_results['accuracy']*100:.2f}%)")
    
    # 保存评估指标
    manager.save_evaluation_metrics({
        'accuracy': src_results['accuracy'],
        'correct': src_results['correct'],
        'total': src_results['total']
    }, 'source_test')
    
    manager.save_evaluation_metrics({
        'accuracy': tgt_results['accuracy'],
        'correct': tgt_results['correct'],
        'total': tgt_results['total']
    }, 'target_test')
    
    # 保存模型
    manager.save_model(model, 'baseline_model')
    
    # 生成混淆矩阵
    manager.save_confusion_matrix(tgt_results['labels'], tgt_results['predictions'], 
                                  class_names=[str(i) for i in range(10)])
    
    # 生成实验报告
    report_path = manager.generate_report()
    print(f"\n实验报告已生成: {report_path}")
    
    # 保存结果摘要
    summary_path = manager.save_summary()
    print(f"结果摘要已保存: {summary_path}")
    
    print(f"\n实验完成！所有结果已保存到: {manager.get_experiment_dir()}")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    main()