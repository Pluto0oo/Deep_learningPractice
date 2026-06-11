"""
小样本学习评分函数对比实验 - 简化版（快速测试）
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, average_precision_score

# 设置环境变量解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建输出文件夹
output_dir = 'few_shot_real_plots'
os.makedirs(output_dir, exist_ok=True)

# 设置Matplotlib科研论文风格
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'font.family': 'sans-serif',
    'axes.unicode_minus': False,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7
})

# ==================== OmniCNN特征提取器 ====================
class OmniCNN(nn.Module):
    def __init__(self, in_channels=1, embedding_dim=64):
        super(OmniCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.embedding = nn.Linear(64, embedding_dim)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return F.normalize(x, dim=-1)

# ==================== 评分函数 ====================
class CosineSimilarity:
    def __init__(self): self.name = "余弦相似度"
    def compute(self, query_features, prototypes):
        return torch.matmul(query_features, prototypes.transpose(0, 1))

class ProtoNetDistance:
    def __init__(self, scale=10.0):
        self.name = f"ProtoNet (s={scale})"
        self.scale = scale
    def compute(self, query_features, prototypes):
        diff = query_features.unsqueeze(1) - prototypes.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        return -self.scale * dist

class CosFace:
    def __init__(self, s=10.0, m=0.1):
        self.name = f"CosFace (s={s}, m={m})"
        self.s = s
        self.m = m
    def compute(self, query_features, prototypes):
        cosine = torch.matmul(query_features, prototypes.transpose(0, 1))
        return self.s * (cosine - self.m)

class ArcFace:
    def __init__(self, s=10.0, m=0.2):
        self.name = f"ArcFace (s={s}, m={m})"
        self.s = s
        self.m = m
    def compute(self, query_features, prototypes):
        cosine = torch.matmul(query_features, prototypes.transpose(0, 1))
        theta = torch.acos(torch.clamp(cosine, -1+1e-7, 1-1e-7))
        return self.s * torch.cos(theta + self.m)

# ==================== 数据集 ====================
def create_tasks(num_classes=20, num_support=5, num_query=10, num_tasks=200):
    """创建小样本学习任务"""
    tasks = []
    for _ in range(num_tasks):
        # 生成类原型
        class_prototypes = np.random.randn(num_classes, 1, 28, 28) * 0.5
        
        support_set = []
        support_labels = []
        query_set = []
        query_labels = []
        
        for c in range(num_classes):
            # 支持样本
            support = class_prototypes[c] + np.random.randn(num_support, 1, 28, 28) * 0.3
            support_set.append(support)
            support_labels.extend([c] * num_support)
            
            # 查询样本
            query = class_prototypes[c] + np.random.randn(num_query, 1, 28, 28) * 0.5
            query_set.append(query)
            query_labels.extend([c] * num_query)
        
        tasks.append({
            'support': np.concatenate(support_set, axis=0),
            'support_labels': np.array(support_labels),
            'query': np.concatenate(query_set, axis=0),
            'query_labels': np.array(query_labels)
        })
    return tasks

# ==================== 训练 ====================
def train_method(feature_extractor, scoring_fn, train_tasks, test_tasks, epochs=10):
    """训练单个评分函数"""
    optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        feature_extractor.train()
        epoch_loss = []
        epoch_acc = []
        
        for task in train_tasks[:30]:  # 快速训练
            support = torch.tensor(task['support'], dtype=torch.float32).to(device)
            support_labels = torch.tensor(task['support_labels'], dtype=torch.long).to(device)
            query = torch.tensor(task['query'], dtype=torch.float32).to(device)
            query_labels = torch.tensor(task['query_labels'], dtype=torch.long).to(device)
            
            # 提取特征
            support_features = feature_extractor(support)
            query_features = feature_extractor(query)
            
            # 计算原型
            prototypes = []
            for label in torch.unique(support_labels):
                mask = support_labels == label
                proto = support_features[mask].mean(dim=0)
                prototypes.append(proto)
            prototypes = torch.stack(prototypes)
            
            # 计算分数
            scores = scoring_fn.compute(query_features, prototypes)
            loss = F.cross_entropy(scores, query_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(scores, dim=-1)
            acc = (preds == query_labels).float().mean().item()
            epoch_loss.append(loss.item())
            epoch_acc.append(acc)
        
        # 测试
        feature_extractor.eval()
        test_accs = []
        with torch.no_grad():
            for task in test_tasks:
                support = torch.tensor(task['support'], dtype=torch.float32).to(device)
                support_labels = torch.tensor(task['support_labels'], dtype=torch.long).to(device)
                query = torch.tensor(task['query'], dtype=torch.float32).to(device)
                query_labels = torch.tensor(task['query_labels'], dtype=torch.long).to(device)
                
                support_features = feature_extractor(support)
                query_features = feature_extractor(query)
                
                prototypes = []
                for label in torch.unique(support_labels):
                    mask = support_labels == label
                    proto = support_features[mask].mean(dim=0)
                    prototypes.append(proto)
                prototypes = torch.stack(prototypes)
                
                scores = scoring_fn.compute(query_features, prototypes)
                preds = torch.argmax(scores, dim=-1)
                acc = accuracy_score(query_labels.cpu(), preds.cpu())
                test_accs.append(acc)
        
        print(f"{scoring_fn.name} - Epoch {epoch+1}/{epochs}: "
              f"Train Acc={np.mean(epoch_acc):.4f}, Test Acc={np.mean(test_accs):.4f}")
    
    return np.mean(epoch_acc), np.mean(test_accs)

# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("小样本学习评分函数对比实验")
    print("数据集: 合成Omniglot风格数据")
    print("=" * 60)
    
    # 创建数据集
    print("\n创建数据集...")
    train_tasks = create_tasks(num_tasks=200)
    test_tasks = create_tasks(num_tasks=100)
    print(f"训练任务: {len(train_tasks)}, 测试任务: {len(test_tasks)}")
    
    # 评分函数
    scoring_functions = [
        CosineSimilarity(),
        ProtoNetDistance(scale=10.0),
        CosFace(s=10.0, m=0.1),
        ArcFace(s=10.0, m=0.2),
    ]
    
    results = {}
    
    # 训练每个评分函数
    for sf in scoring_functions:
        print(f"\n训练: {sf.name}")
        feature_extractor = OmniCNN(in_channels=1, embedding_dim=64)
        feature_extractor.to(device)
        train_acc, test_acc = train_method(feature_extractor, sf, train_tasks, test_tasks)
        results[sf.name] = {'train_acc': train_acc, 'test_acc': test_acc}
    
    # 可视化
    print("\n生成可视化图表...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    test_accs = [results[n]['test_acc'] for n in names]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    
    bars = ax.bar(range(len(names)), test_accs, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, acc in zip(bars, test_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_title('小样本学习评分函数性能对比', fontweight='bold')
    ax.set_ylabel('测试准确率')
    ax.set_ylim(0, max(test_accs) * 1.15)
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'few_shot_real_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n图表已保存至 {output_dir}/")
    
    # 排序输出
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True)
    print("\n【最终排名】:")
    for i, (name, res) in enumerate(sorted_results):
        print(f"{i+1}. {name}: {res['test_acc']:.4f}")

if __name__ == '__main__':
    main()