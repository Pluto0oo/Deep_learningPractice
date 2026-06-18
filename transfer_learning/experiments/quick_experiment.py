"""快速对比实验 - 完整流程"""
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
print("迁移学习评估对比实验")
print("实验任务: MNIST -> MNIST-M")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}\n")

start_time = time.time()

print("加载数据...")
mnist_dir = 'data/mnist/MNIST/raw/'

with gzip.open(mnist_dir + 'train-images-idx3-ubyte.gz', 'rb') as f:
    src_train_imgs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)[:10000]
with gzip.open(mnist_dir + 'train-labels-idx1-ubyte.gz', 'rb') as f:
    src_train_labels = np.frombuffer(f.read(), np.uint8, offset=8)[:10000]
with gzip.open(mnist_dir + 't10k-images-idx3-ubyte.gz', 'rb') as f:
    tgt_test_imgs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)[:2000]
with gzip.open(mnist_dir + 't10k-labels-idx1-ubyte.gz', 'rb') as f:
    tgt_test_labels = np.frombuffer(f.read(), np.uint8, offset=8)[:2000]

src_train_imgs = src_train_imgs.astype(np.float32) / 255.0
src_train_imgs = np.repeat(src_train_imgs[:, np.newaxis, :, :], 3, axis=1)
tgt_test_imgs = tgt_test_imgs.astype(np.float32) / 255.0
tgt_test_imgs = np.repeat(tgt_test_imgs[:, np.newaxis, :, :], 3, axis=1)

with open('data/mnistm/mnistm_data.pkl', 'rb') as f:
    mnistm_data = pickle.load(f)

tgt_train_imgs = mnistm_data['train'].astype(np.float32)[:10000] / 255.0
tgt_train_labels = mnistm_data['train_labels'][:10000]
tgt_test_imgs_actual = mnistm_data['test'].astype(np.float32)[:2000] / 255.0
tgt_test_labels_actual = mnistm_data['test_labels'][:2000]

print(f"源域训练数据: {src_train_imgs.shape}")
print(f"目标域训练数据: {tgt_train_imgs.shape}")
print(f"目标域测试数据: {tgt_test_imgs_actual.shape}")

def get_loader(images, labels, batch_size=64, shuffle=True):
    images_copy = images.copy()
    labels_copy = labels.copy()
    if images_copy.ndim == 4 and images_copy.shape[3] == 3:
        images_copy = np.transpose(images_copy, (0, 3, 1, 2))
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(images_copy), torch.from_numpy(labels_copy))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

src_loader = get_loader(src_train_imgs, src_train_labels)
tgt_train_loader = get_loader(tgt_train_imgs, tgt_train_labels)
tgt_test_loader = get_loader(tgt_test_imgs_actual, tgt_test_labels_actual)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(48 * 7 * 7, num_classes)
    
    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))

def train_model(model, train_loader, device, epochs=10, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
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
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {correct/total:.4f}")
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

print("\n" + "=" * 60)
print("实验1: 基线模型 (仅源域训练)")
print("=" * 60)
baseline_model = SimpleCNN().to(device)
baseline_model = train_model(baseline_model, src_loader, device, epochs=10)
baseline_acc = evaluate(baseline_model, tgt_test_loader, device)
print(f"基线模型目标域准确率: {baseline_acc:.4f}")

print("\n" + "=" * 60)
print("实验2: 微调模型 (源域预训练 + 目标域微调)")
print("=" * 60)
finetune_model = SimpleCNN().to(device)
finetune_model = train_model(finetune_model, src_loader, device, epochs=10)
finetune_model = train_model(finetune_model, tgt_train_loader, device, epochs=5, lr=1e-4)
finetune_acc = evaluate(finetune_model, tgt_test_loader, device)
print(f"微调模型目标域准确率: {finetune_acc:.4f}")

print("\n" + "=" * 60)
print("实验3: DANN模型 (域对抗神经网络)")
print("=" * 60)

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
            nn.Dropout(0.5),
            nn.Linear(100, 2)
        )
    
    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x).view(x.size(0), -1)
        class_output = self.label_classifier(features)
        domain_output = self.domain_classifier(GradientReversalLayer.apply(features, alpha))
        return class_output, domain_output

def train_dann(model, src_loader, tgt_loader, device, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_loader)
    num_batches = min(len(src_loader), len(tgt_loader))
    
    for epoch in range(epochs):
        total_class_loss = 0
        total_domain_loss = 0
        correct = 0
        total = 0
        
        for i in range(num_batches):
            p = float(i + epoch * num_batches) / (epochs * num_batches)
            alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1
            
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
            
            optimizer.zero_grad()
            
            src_class_out, src_domain_out = model(src_imgs.to(device), alpha)
            tgt_class_out, tgt_domain_out = model(tgt_imgs.to(device), alpha)
            
            class_loss = criterion(src_class_out, src_labels.to(device))
            src_domain_loss = criterion(src_domain_out, src_domain_labels)
            tgt_domain_loss = criterion(tgt_domain_out, tgt_domain_labels)
            domain_loss = src_domain_loss + tgt_domain_loss
            
            loss = class_loss + 0.1 * domain_loss
            loss.backward()
            optimizer.step()
            
            total_class_loss += class_loss.item()
            total_domain_loss += domain_loss.item()
            _, pred = torch.max(src_class_out, 1)
            correct += (pred == src_labels.to(device)).sum().item()
            total += src_labels.size(0)
        
        print(f"DANN Epoch {epoch+1}/{epochs} | Class Loss: {total_class_loss/num_batches:.4f} | Domain Loss: {total_domain_loss/num_batches:.4f} | Src Acc: {correct/total:.4f}")
    
    return model

dann_model = DANN().to(device)
dann_model = train_dann(dann_model, src_loader, tgt_train_loader, device, epochs=10)
dann_acc = evaluate(dann_model, tgt_test_loader, device)
print(f"DANN模型目标域准确率: {dann_acc:.4f}")

total_time = time.time() - start_time

print("\n" + "=" * 60)
print("实验结果汇总")
print("=" * 60)
print(f"基线模型准确率: {baseline_acc:.4f}")
print(f"微调模型准确率: {finetune_acc:.4f}")
print(f"DANN模型准确率: {dann_acc:.4f}")
print(f"\n总运行时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")

os.makedirs('results', exist_ok=True)
with open('results/experiment_results.txt', 'w') as f:
    f.write("迁移学习评估对比实验结果\n")
    f.write("=" * 40 + "\n")
    f.write(f"实验任务: MNIST -> MNIST-M\n")
    f.write(f"基线模型准确率: {baseline_acc:.4f}\n")
    f.write(f"微调模型准确率: {finetune_acc:.4f}\n")
    f.write(f"DANN模型准确率: {dann_acc:.4f}\n")
    f.write(f"总运行时间: {total_time:.2f}秒\n")
print("\n结果已保存到 results/experiment_results.txt")
