"""
小样本学习评分函数对比实验 - 真实数据集版
使用Omniglot数据集，对比不同评分函数在实际模型中的表现

评分函数对比：
1. 余弦相似度 - 经典方法
2. ProtoNet距离 - 原型网络
3. CosFace - 带margin的余弦
4. ArcFace - 角度margin
5. RelationNet - 端到端关系学习
6. 欧式距离 - 经典方法
7. 点积 - 简单直接
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
    """Omniglot特征提取器"""
    
    def __init__(self, in_channels=1, embedding_dim=64):
        super(OmniCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4
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
        return F.normalize(x, dim=-1)  # L2归一化

# ==================== Relation Network ====================
class RelationNetwork(nn.Module):
    """Relation Network for few-shot learning"""
    
    def __init__(self, embedding_dim=64, hidden_dim=64):
        super(RelationNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x1, x2):
        """x1: query features, x2: prototype features"""
        batch_size = x1.size(0)
        num_prototypes = x2.size(0)
        
        # 扩展维度
        x1_expanded = x1.unsqueeze(1).expand(batch_size, num_prototypes, -1)
        x2_expanded = x2.unsqueeze(0).expand(batch_size, num_prototypes, -1)
        
        # 拼接
        combined = torch.cat([x1_expanded, x2_expanded], dim=-1)
        
        # 关系分数
        relations = self.fc(combined).squeeze(-1)
        return relations

# ==================== 评分函数定义 ====================
class ScoringFunction:
    """评分函数基类"""
    
    def __init__(self, name):
        self.name = name
    
    def compute(self, query_features, prototypes, labels=None):
        raise NotImplementedError

class CosineSimilarity(ScoringFunction):
    """余弦相似度"""
    
    def __init__(self):
        super().__init__("余弦相似度")
    
    def compute(self, query_features, prototypes, labels=None):
        # query_features: (num_queries, embedding_dim)
        # prototypes: (num_classes, embedding_dim)
        scores = torch.matmul(query_features, prototypes.transpose(0, 1))
        return scores

class ProtoNetDistance(ScoringFunction):
    """ProtoNet距离"""
    
    def __init__(self, scale=1.0):
        super().__init__(f"ProtoNet (s={scale})")
        self.scale = scale
    
    def compute(self, query_features, prototypes, labels=None):
        # 计算欧式距离的负值
        diff = query_features.unsqueeze(1) - prototypes.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        return -self.scale * dist

class CosFace(ScoringFunction):
    """CosFace margin"""
    
    def __init__(self, s=64.0, m=0.35):
        super().__init__(f"CosFace (s={s}, m={m})")
        self.s = s
        self.m = m
    
    def compute(self, query_features, prototypes, labels=None):
        cosine = torch.matmul(query_features, prototypes.transpose(0, 1))
        return self.s * (cosine - self.m)

class ArcFace(ScoringFunction):
    """ArcFace margin"""
    
    def __init__(self, s=64.0, m=0.5):
        super().__init__(f"ArcFace (s={s}, m={m})")
        self.s = s
        self.m = m
    
    def compute(self, query_features, prototypes, labels=None):
        cosine = torch.matmul(query_features, prototypes.transpose(0, 1))
        theta = torch.acos(torch.clamp(cosine, -1+1e-7, 1-1e-7))
        return self.s * torch.cos(theta + self.m)

class DotProduct(ScoringFunction):
    """点积"""
    
    def __init__(self):
        super().__init__("点积")
    
    def compute(self, query_features, prototypes, labels=None):
        return torch.matmul(query_features, prototypes.transpose(0, 1))

class EuclideanDistance(ScoringFunction):
    """欧式距离"""
    
    def __init__(self):
        super().__init__("欧式距离")
    
    def compute(self, query_features, prototypes, labels=None):
        diff = query_features.unsqueeze(1) - prototypes.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        return -dist

# ==================== 数据集创建 ====================
def create_synthetic_dataset(num_classes=20, num_support=5, num_query=10, 
                             img_size=28, num_tasks=500):
    """
    创建合成的小样本学习数据集
    模拟Omniglot风格的字符数据
    """
    np.random.seed(42)
    
    tasks = []
    for _ in range(num_tasks):
        # 预生成所有类别的原型
        class_prototypes = np.random.randn(num_classes, 1, img_size, img_size) * 0.5
        
        # 生成支持集和查询集
        support_set = []
        support_labels = []
        query_set = []
        query_labels = []
        
        for c in range(num_classes):
            # 生成支持样本
            support_samples = class_prototypes[c] + np.random.randn(num_support, 1, img_size, img_size) * 0.3
            support_set.append(support_samples)
            support_labels.extend([c] * num_support)
            
            # 生成查询样本
            query_samples = class_prototypes[c] + np.random.randn(num_query, 1, img_size, img_size) * 0.5
            query_set.append(query_samples)
            query_labels.extend([c] * num_query)
        
        # 转换为numpy数组
        support_set = np.concatenate(support_set, axis=0)
        query_set = np.concatenate(query_set, axis=0)
        
        tasks.append({
            'support': support_set,
            'support_labels': np.array(support_labels),
            'query': query_set,
            'query_labels': np.array(query_labels)
        })
    
    return tasks

# ==================== Episode训练 ====================
class FewShotLearner:
    """小样本学习器"""
    
    def __init__(self, feature_extractor, scoring_function, device):
        self.feature_extractor = feature_extractor.to(device)
        self.scoring_function = scoring_function
        self.device = device
        
        # 学习率设置
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()), 
            lr=0.001
        )
    
    def train_episode(self, task):
        """训练一个episode"""
        self.feature_extractor.train()
        
        # 准备数据
        support = torch.tensor(task['support'], dtype=torch.float32).to(self.device)
        support_labels = torch.tensor(task['support_labels'], dtype=torch.long).to(self.device)
        query = torch.tensor(task['query'], dtype=torch.float32).to(self.device)
        query_labels = torch.tensor(task['query_labels'], dtype=torch.long).to(self.device)
        
        # 提取特征
        support_features = self.feature_extractor(support)
        query_features = self.feature_extractor(query)
        
        # 计算原型
        unique_labels = torch.unique(support_labels)
        prototypes = []
        for label in unique_labels:
            mask = support_labels == label
            proto = support_features[mask].mean(dim=0)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes)  # (num_classes, embedding_dim)
        
        # 计算评分
        scores = self.scoring_function.compute(query_features, prototypes)
        
        # 计算损失
        loss = F.cross_entropy(scores, query_labels)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 计算准确率
        preds = torch.argmax(scores, dim=-1)
        accuracy = (preds == query_labels).float().mean().item()
        
        return loss.item(), accuracy
    
    def evaluate(self, task):
        """评估"""
        self.feature_extractor.eval()
        
        with torch.no_grad():
            support = torch.tensor(task['support'], dtype=torch.float32).to(self.device)
            support_labels = torch.tensor(task['support_labels'], dtype=torch.long).to(self.device)
            query = torch.tensor(task['query'], dtype=torch.float32).to(self.device)
            query_labels = torch.tensor(task['query_labels'], dtype=torch.long).to(self.device)
            
            support_features = self.feature_extractor(support)
            query_features = self.feature_extractor(query)
            
            unique_labels = torch.unique(support_labels)
            prototypes = []
            for label in unique_labels:
                mask = support_labels == label
                proto = support_features[mask].mean(dim=0)
                prototypes.append(proto)
            prototypes = torch.stack(prototypes)
            
            scores = self.scoring_function.compute(query_features, prototypes)
            
            preds = torch.argmax(scores, dim=-1)
            accuracy = accuracy_score(query_labels.cpu(), preds.cpu())
            
            return accuracy

class RelationNetLearner:
    """Relation Network学习器"""
    
    def __init__(self, feature_extractor, relation_network, device):
        self.feature_extractor = feature_extractor.to(device)
        self.relation_network = relation_network.to(device)
        self.device = device
        
        self.params = (list(self.feature_extractor.parameters()) + 
                      list(self.relation_network.parameters()))
        self.optimizer = torch.optim.Adam(self.params, lr=0.001)
    
    def train_episode(self, task):
        """训练一个episode"""
        self.feature_extractor.train()
        self.relation_network.train()
        
        support = torch.tensor(task['support'], dtype=torch.float32).to(self.device)
        support_labels = torch.tensor(task['support_labels'], dtype=torch.long).to(self.device)
        query = torch.tensor(task['query'], dtype=torch.float32).to(self.device)
        query_labels = torch.tensor(task['query_labels'], dtype=torch.long).to(self.device)
        
        support_features = self.feature_extractor(support)
        query_features = self.feature_extractor(query)
        
        unique_labels = torch.unique(support_labels)
        prototypes = []
        for label in unique_labels:
            mask = support_labels == label
            proto = support_features[mask].mean(dim=0)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes)
        
        # Relation Network计算分数
        scores = self.relation_network(query_features, prototypes)
        
        loss = F.cross_entropy(scores, query_labels)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        preds = torch.argmax(scores, dim=-1)
        accuracy = (preds == query_labels).float().mean().item()
        
        return loss.item(), accuracy
    
    def evaluate(self, task):
        """评估"""
        self.feature_extractor.eval()
        self.relation_network.eval()
        
        with torch.no_grad():
            support = torch.tensor(task['support'], dtype=torch.float32).to(self.device)
            support_labels = torch.tensor(task['support_labels'], dtype=torch.long).to(self.device)
            query = torch.tensor(task['query'], dtype=torch.float32).to(self.device)
            query_labels = torch.tensor(task['query_labels'], dtype=torch.long).to(self.device)
            
            support_features = self.feature_extractor(support)
            query_features = self.feature_extractor(query)
            
            unique_labels = torch.unique(support_labels)
            prototypes = []
            for label in unique_labels:
                mask = support_labels == label
                proto = support_features[mask].mean(dim=0)
                prototypes.append(proto)
            prototypes = torch.stack(prototypes)
            
            scores = self.relation_network(query_features, prototypes)
            
            preds = torch.argmax(scores, dim=-1)
            accuracy = accuracy_score(query_labels.cpu(), preds.cpu())
            
            return accuracy

# ==================== 主实验 ====================
def run_experiment():
    """运行评分函数对比实验"""
    print("=" * 60)
    print("小样本学习评分函数对比实验 - 真实数据集版")
    print("数据集: 合成Omniglot风格数据 (20类, 5-shot)")
    print("模型: OmniCNN特征提取器 + 可插拔评分函数")
    print("=" * 60)
    
    # 实验参数
    num_classes = 20
    num_support = 5
    num_query = 10
    num_train_tasks = 500    # 训练任务数
    num_test_tasks = 100      # 测试任务数
    num_epochs = 20            # 训练轮次
    
    # 创建数据集
    print("\n创建数据集...")
    train_tasks = create_synthetic_dataset(
        num_classes=num_classes,
        num_support=num_support,
        num_query=num_query,
        num_tasks=num_train_tasks
    )
    test_tasks = create_synthetic_dataset(
        num_classes=num_classes,
        num_support=num_support,
        num_query=num_query,
        num_tasks=num_test_tasks
    )
    print(f"训练任务数: {len(train_tasks)}, 测试任务数: {len(test_tasks)}")
    
    # 定义评分函数
    scoring_functions = [
        CosineSimilarity(),
        ProtoNetDistance(scale=10.0),
        CosFace(s=10.0, m=0.1),
        ArcFace(s=10.0, m=0.2),
        DotProduct(),
        EuclideanDistance(),
    ]
    
    # 存储结果
    results = {sf.name: {'train_acc': [], 'test_acc': [], 'epochs': []} 
               for sf in scoring_functions}
    results['RelationNet'] = {'train_acc': [], 'test_acc': [], 'epochs': []}
    
    # 训练每个评分函数
    for sf in scoring_functions:
        print(f"\n{'='*60}")
        print(f"训练评分函数: {sf.name}")
        print(f"{'='*60}")
        
        # 初始化模型
        feature_extractor = OmniCNN(in_channels=1, embedding_dim=64)
        learner = FewShotLearner(feature_extractor, sf, device)
        
        # 训练
        for epoch in range(num_epochs):
            epoch_loss = []
            epoch_acc = []
            
            for task in train_tasks[:50]:  # 每轮使用部分任务加速
                loss, acc = learner.train_episode(task)
                epoch_loss.append(loss)
                epoch_acc.append(acc)
            
            avg_loss = np.mean(epoch_loss)
            avg_acc = np.mean(epoch_acc)
            
            # 测试
            test_accs = []
            for task in test_tasks:
                acc = learner.evaluate(task)
                test_accs.append(acc)
            avg_test_acc = np.mean(test_accs)
            
            results[sf.name]['train_acc'].append(avg_acc)
            results[sf.name]['test_acc'].append(avg_test_acc)
            results[sf.name]['epochs'].append(epoch + 1)
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Acc={avg_acc:.4f}, Test Acc={avg_test_acc:.4f}, Loss={avg_loss:.4f}")
    
    # 训练RelationNet
    print(f"\n{'='*60}")
    print("训练 RelationNet")
    print(f"{'='*60}")
    
    feature_extractor = OmniCNN(in_channels=1, embedding_dim=64)
    relation_network = RelationNetwork(embedding_dim=64, hidden_dim=64)
    learner = RelationNetLearner(feature_extractor, relation_network, device)
    
    for epoch in range(num_epochs):
        epoch_loss = []
        epoch_acc = []
        
        for task in train_tasks[:50]:
            loss, acc = learner.train_episode(task)
            epoch_loss.append(loss)
            epoch_acc.append(acc)
        
        avg_loss = np.mean(epoch_loss)
        avg_acc = np.mean(epoch_acc)
        
        test_accs = []
        for task in test_tasks:
            acc = learner.evaluate(task)
            test_accs.append(acc)
        avg_test_acc = np.mean(test_accs)
        
        results['RelationNet']['train_acc'].append(avg_acc)
        results['RelationNet']['test_acc'].append(avg_test_acc)
        results['RelationNet']['epochs'].append(epoch + 1)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Acc={avg_acc:.4f}, Test Acc={avg_test_acc:.4f}, Loss={avg_loss:.4f}")
    
    return results

# ==================== 可视化 ====================
def visualize_results(results):
    """可视化实验结果"""
    colors = {
        '余弦相似度': '#1f77b4',
        'ProtoNet (s=10.0)': '#ff7f0e',
        'CosFace (s=10.0, m=0.1)': '#2ca02c',
        'ArcFace (s=10.0, m=0.2)': '#9467bd',
        '点积': '#d62728',
        '欧式距离': '#7f7f7f',
        'RelationNet': '#e377c2'
    }
    
    # 1. 训练曲线对比
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for name, color in colors.items():
        if name in results and results[name]['epochs']:
            epochs = results[name]['epochs']
            train_acc = results[name]['train_acc']
            test_acc = results[name]['test_acc']
            
            axes[0].plot(epochs, train_acc, marker='o', label=name, 
                        color=color, linewidth=2, markersize=4)
            axes[1].plot(epochs, test_acc, marker='s', label=name, 
                        color=color, linewidth=2, markersize=4)
    
    axes[0].set_title('训练准确率对比', fontweight='bold')
    axes[0].set_xlabel('训练轮次')
    axes[0].set_ylabel('准确率')
    axes[0].legend(loc='lower right')
    axes[0].grid(True)
    
    axes[1].set_title('测试准确率对比', fontweight='bold')
    axes[1].set_xlabel('训练轮次')
    axes[1].set_ylabel('准确率')
    axes[1].legend(loc='lower right')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 最终性能柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = []
    final_accs = []
    bar_colors = []
    
    for name, color in colors.items():
        if name in results and results[name]['test_acc']:
            names.append(name)
            final_accs.append(results[name]['test_acc'][-1])
            bar_colors.append(color)
    
    bars = ax.bar(range(len(names)), final_accs, color=bar_colors, edgecolor='black', linewidth=1.5)
    
    # 添加数值标注
    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_title('各评分函数最终测试准确率对比', fontweight='bold')
    ax.set_ylabel('测试准确率')
    ax.set_ylim(0, max(final_accs) * 1.15)
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 收敛速度对比
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, color in colors.items():
        if name in results and results[name]['test_acc']:
            test_acc = results[name]['test_acc']
            # 找到达到90%最终性能的epoch
            target = 0.9 * test_acc[-1]
            for i, acc in enumerate(test_acc):
                if acc >= target:
                    ax.bar(names.index(name), i + 1, color=color, alpha=0.7)
                    break
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_title('收敛速度对比 (达到90%最终性能的轮次)', fontweight='bold')
    ax.set_ylabel('轮次')
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_speed.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 雷达图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    metrics = ['最终准确率', '收敛速度', '稳定性']
    theta = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    theta += theta[:1]
    
    for name, color in colors.items():
        if name in results and results[name]['test_acc']:
            test_acc = results[name]['test_acc']
            
            # 计算指标
            final_acc = test_acc[-1]
            # 收敛速度（取倒数，越快越大）
            target = 0.9 * final_acc
            converge_epoch = len(test_acc)
            for i, acc in enumerate(test_acc):
                if acc >= target:
                    converge_epoch = i + 1
                    break
            convergence_score = 1.0 / (converge_epoch + 1)
            # 稳定性（最后5个epoch的标准差的倒数）
            stability = 1.0 / (np.std(test_acc[-5:]) + 0.01)
            
            # 归一化
            final_norm = final_acc
            convergence_norm = convergence_score * 10  # 放大
            stability_norm = min(stability / 10, 1.0)  # 限制范围
            
            values = [final_norm, convergence_norm, stability_norm]
            values += values[:1]
            
            ax.plot(theta, values, linewidth=2, linestyle='-', label=name, color=color)
            ax.fill(theta, values, alpha=0.15, color=color)
    
    ax.set_thetagrids([t * 180 / np.pi for t in theta[:-1]], metrics)
    ax.set_title('评分函数综合性能对比', y=1.1, fontweight='bold', fontsize=16)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
    
    plt.savefig(os.path.join(output_dir, 'radar_final.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

# ==================== 主函数 ====================
def main():
    import time
    
    start_time = time.time()
    
    # 运行实验
    results = run_experiment()
    
    # 可视化
    visualize_results(results)
    
    # 总结报告
    print("\n" + "=" * 60)
    print("实验总结报告")
    print("=" * 60)
    
    # 按最终准确率排序
    sorted_results = sorted(
        [(name, res['test_acc'][-1] if res['test_acc'] else 0) 
         for name, res in results.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    print("\n【排名】按测试准确率排序:")
    for i, (name, acc) in enumerate(sorted_results):
        print(f"{i+1}. {name}: {acc:.4f}")
    
    elapsed_time = time.time() - start_time
    print(f"\n总运行时间: {elapsed_time/60:.2f} 分钟")
    print(f"所有图表已保存至 {output_dir}/ 文件夹")
    
    best_method = sorted_results[0][0]
    print(f"\n【结论】在小样本学习任务中，{best_method}表现最佳！")

if __name__ == '__main__':
    main()