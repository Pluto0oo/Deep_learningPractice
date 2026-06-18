"""优化后的DANN模型训练 - 详细版"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gzip
import pickle
import os
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("优化后的DANN模型训练")
print("实验任务: MNIST -> MNIST-M")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}\n")

# 加载数据
print("加载数据...")
mnist_dir = 'data/mnist/MNIST/raw/'

with gzip.open(mnist_dir + 'train-images-idx3-ubyte.gz', 'rb') as f:
    src_train_imgs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)[:5000]
with gzip.open(mnist_dir + 'train-labels-idx1-ubyte.gz', 'rb') as f:
    src_train_labels = np.frombuffer(f.read(), np.uint8, offset=8)[:5000]

src_train_imgs = src_train_imgs.astype(np.float32) / 255.0
src_train_imgs = np.repeat(src_train_imgs[:, np.newaxis, :, :], 3, axis=1)
print(f"源域训练数据: {src_train_imgs.shape}")

with open('data/mnistm/mnistm_data.pkl', 'rb') as f:
    mnistm_data = pickle.load(f)

tgt_train_imgs = mnistm_data['train'].astype(np.float32)[:5000] / 255.0
tgt_train_labels = mnistm_data['train_labels'][:5000]
tgt_test_imgs = mnistm_data['test'].astype(np.float32)[:1000] / 255.0
tgt_test_labels = mnistm_data['test_labels'][:1000]

print(f"目标域训练数据: {tgt_train_imgs.shape}")
print(f"目标域测试数据: {tgt_test_imgs.shape}")

def get_loader(images, labels, batch_size=64, shuffle=True):
    images_copy = images.copy()
    labels_copy = labels.copy()
    if images_copy.ndim == 4 and images_copy.shape[3] == 3:
        images_copy = np.transpose(images_copy, (0, 3, 1, 2))
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(images_copy), torch.from_numpy(labels_copy))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

src_loader = get_loader(src_train_imgs, src_train_labels, batch_size=32)
tgt_train_loader = get_loader(tgt_train_imgs, tgt_train_labels, batch_size=32)
tgt_test_loader = get_loader(tgt_test_imgs, tgt_test_labels, batch_size=32)

print(f"\n源域数据加载器: {len(src_loader)} batches")
print(f"目标域训练加载器: {len(tgt_train_loader)} batches")
print(f"目标域测试加载器: {len(tgt_test_loader)} batches")

# 优化后的DANN模型
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DANN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.label_classifier = nn.Linear(48 * 7 * 7, num_classes)
        self.domain_classifier = nn.Sequential(
            nn.Linear(48 * 7 * 7, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2)
        )
    
    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x).view(x.size(0), -1)
        class_output = self.label_classifier(features)
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain_output = self.domain_classifier(reversed_features)
        return class_output, domain_output

def train_dann_optimized(model, src_loader, tgt_loader, device, epochs=10, lr=1e-3, domain_weight=1.0):
    print(f"\n开始训练: epochs={epochs}, lr={lr}, domain_weight={domain_weight}")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    
    model.to(device)
    model.train()
    
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_loader)
    num_batches = min(len(src_loader), len(tgt_loader))
    
    for epoch in range(epochs):
        total_class_loss = 0
        total_domain_loss = 0
        correct_class = 0
        correct_domain = 0
        total = 0
        
        for i in range(num_batches):
            try:
                src_imgs, src_labels = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                src_imgs, src_labels = next(src_iter)
            
            try:
                tgt_imgs, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_imgs, _ = next(tgt_iter)
            
            src_domain_labels = torch.zeros(src_imgs.size(0)).long().to(device)
            tgt_domain_labels = torch.ones(tgt_imgs.size(0)).long().to(device)
            
            p = float(i + epoch * num_batches) / (epochs * num_batches)
            alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1
            
            optimizer.zero_grad()
            
            src_class_out, src_domain_out = model(src_imgs.to(device), alpha)
            tgt_class_out, tgt_domain_out = model(tgt_imgs.to(device), alpha)
            
            class_loss = criterion_class(src_class_out, src_labels.to(device))
            src_domain_loss = criterion_domain(src_domain_out, src_domain_labels)
            tgt_domain_loss = criterion_domain(tgt_domain_out, tgt_domain_labels)
            domain_loss = src_domain_loss + tgt_domain_loss
            
            loss = class_loss + domain_weight * domain_loss
            loss.backward()
            optimizer.step()
            
            total_class_loss += class_loss.item()
            total_domain_loss += domain_loss.item()
            _, pred_class = torch.max(src_class_out, 1)
            correct_class += (pred_class == src_labels.to(device)).sum().item()
            _, pred_domain_src = torch.max(src_domain_out, 1)
            _, pred_domain_tgt = torch.max(tgt_domain_out, 1)
            correct_domain += (pred_domain_src == src_domain_labels).sum().item()
            correct_domain += (pred_domain_tgt == tgt_domain_labels).sum().item()
            total += src_labels.size(0)
        
        domain_acc = correct_domain / (2 * total)
        print(f"Epoch {epoch+1}/{epochs} | Class Loss: {total_class_loss/num_batches:.4f} | Domain Loss: {total_domain_loss/num_batches:.4f} | Src Acc: {correct_class/total:.4f} | Domain Acc: {domain_acc:.4f}")
    
    return model

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total

start_time = time.time()

print("\n" + "=" * 60)
print("训练优化后的DANN模型")
print("=" * 60)

dann_model = DANN().to(device)
dann_model = train_dann_optimized(dann_model, src_loader, tgt_train_loader, device, epochs=10, lr=1e-3, domain_weight=0.5)

dann_acc = evaluate(dann_model, tgt_test_loader, device)
total_time = time.time() - start_time

print(f"\n优化后DANN模型目标域准确率: {dann_acc:.4f} ({dann_acc*100:.2f}%)")
print(f"训练时间: {total_time:.2f}秒")

# 保存结果
os.makedirs('results', exist_ok=True)
with open('results/dann_optimized_results.txt', 'w') as f:
    f.write("DANN模型优化实验结果\n")
    f.write("=" * 40 + "\n")
    f.write(f"目标域准确率: {dann_acc:.4f}\n")
    f.write(f"训练时间: {total_time:.2f}秒\n")
    f.write(f"训练轮数: 10\n")
    f.write(f"学习率: 1e-3\n")
    f.write(f"域损失权重: 0.5\n")

print("结果已保存到 results/dann_optimized_results.txt")
