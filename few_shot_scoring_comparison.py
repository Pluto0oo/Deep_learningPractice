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
output_dir = 'few_shot_plots'
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
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7
})

# ==================== 经典评分函数 ====================
class ClassicScoringFunctions:
    """经典评分函数"""
    
    @staticmethod
    def cosine_similarity(x1, x2):
        """余弦相似度 (2000s)"""
        x1_norm = F.normalize(x1, dim=-1)
        x2_norm = F.normalize(x2, dim=-1)
        return torch.matmul(x1_norm, x2_norm.transpose(-2, -1))
    
    @staticmethod
    def euclidean_distance(x1, x2):
        """欧式距离 (经典)"""
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        dist = torch.norm(diff, dim=-1)
        return -dist
    
    @staticmethod
    def dot_product(x1, x2):
        """点积 (经典)"""
        return torch.matmul(x1, x2.transpose(-2, -1))

# ==================== 近年创新评分函数 ====================
class ModernScoringFunctions:
    """近几年创新评分函数"""
    
    @staticmethod
    def prototypical_distance(x1, x2, alpha=1.0):
        """Prototypical Networks (2017) - 原型距离度量"""
        # x2是原型，计算查询样本到原型的距离
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        dist = torch.norm(diff, dim=-1)
        return -alpha * dist  # 取负使越大越相似
    
    @staticmethod
    def cosface_margin(x1, x2, s=64.0, m=0.35):
        """CosFace (2018) - 带margin的余弦相似度"""
        x1_norm = F.normalize(x1, dim=-1)
        x2_norm = F.normalize(x2, dim=-1)
        cos_theta = torch.matmul(x1_norm, x2_norm.transpose(-2, -1))
        # CosFace: s * (cos(theta) - m)
        return s * (cos_theta - m)
    
    @staticmethod
    def arcface_margin(x1, x2, s=64.0, m=0.5):
        """ArcFace (2019) - 角度空间margin"""
        x1_norm = F.normalize(x1, dim=-1)
        x2_norm = F.normalize(x2, dim=-1)
        cos_theta = torch.matmul(x1_norm, x2_norm.transpose(-2, -1))
        theta = torch.acos(torch.clamp(cos_theta, -1+1e-7, 1-1e-7))
        # ArcFace: s * cos(theta + m)
        return s * torch.cos(theta + m)
    
    @staticmethod
    def nca_score(x1, x2, sigma=1.0):
        """NCA (Neighborhood Components Analysis, 2005但仍广泛使用)"""
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        dist = torch.norm(diff, dim=-1)
        # NCA使用softmax归一化的距离
        return torch.exp(-dist ** 2 / (2 * sigma ** 2))
    
    @staticmethod
    def soft_nn(x1, x2, temperature=0.1):
        """Soft Nearest Neighbor (2019)"""
        x1_norm = F.normalize(x1, dim=-1)
        x2_norm = F.normalize(x2, dim=-1)
        similarity = torch.matmul(x1_norm, x2_norm.transpose(-2, -1))
        return similarity / temperature
    
    @staticmethod
    def triplet_margin(x1, x2, margin=1.0):
        """Triplet Margin Loss启发的评分 (2015)"""
        x1_norm = F.normalize(x1, dim=-1)
        x2_norm = F.normalize(x2, dim=-1)
        similarity = torch.matmul(x1_norm, x2_norm.transpose(-2, -1))
        # 最大化同类相似度，最小化异类相似度
        return similarity + margin

# ==================== Relation Networks评分器 ====================
class RelationNetwork(nn.Module):
    """Relation Networks (2017) - 学习样本对之间的关系"""
    
    def __init__(self, input_dim=512, hidden_dim=256):
        super(RelationNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x1, x2):
        """x1: 查询样本, x2: 原型"""
        # 计算所有样本对的关系分数
        batch_size = x1.size(0)
        num_prototypes = x2.size(0)
        
        # 扩展维度
        x1_expanded = x1.unsqueeze(1).expand(-1, num_prototypes, -1)  # (batch, num_proto, dim)
        x2_expanded = x2.unsqueeze(0).expand(batch_size, -1, -1)      # (batch, num_proto, dim)
        
        # 拼接特征
        combined = torch.cat([x1_expanded, x2_expanded], dim=-1)  # (batch, num_proto, 2*dim)
        
        # 计算关系分数
        relations = self.fc(combined).squeeze(-1)  # (batch, num_proto)
        return relations

# ==================== 小样本学习特征提取器 ====================
class FeatureExtractor(nn.Module):
    """简单的特征提取器"""
    
    def __init__(self, input_dim=512, hidden_dim=256):
        super(FeatureExtractor, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ==================== 数据生成 ====================
def generate_few_shot_data(n_ways=5, n_shots=5, n_queries=15, feature_dim=512):
    """生成小样本学习数据集（更高难度）"""
    np.random.seed(42)
    
    # 生成重叠更大的类别原型
    base_prototype = np.random.randn(feature_dim) * 0.3
    prototypes = []
    for i in range(n_ways):
        # 每个类别原型是基础原型加上微小扰动
        prototype = base_prototype + np.random.randn(feature_dim) * 0.15
        prototypes.append(prototype)
    prototypes = np.array(prototypes)
    
    # 生成支持集和查询集
    support_set = []
    support_labels = []
    query_set = []
    query_labels = []
    
    for way in range(n_ways):
        # 生成支持样本（较大噪声）
        for _ in range(n_shots):
            sample = prototypes[way] + np.random.randn(feature_dim) * 0.4
            support_set.append(sample)
            support_labels.append(way)
        
        # 生成查询样本（更大噪声）
        for _ in range(n_queries):
            sample = prototypes[way] + np.random.randn(feature_dim) * 0.6
            query_set.append(sample)
            query_labels.append(way)
    
    return {
        'support': torch.tensor(np.array(support_set), dtype=torch.float32),
        'support_labels': torch.tensor(support_labels, dtype=torch.long),
        'query': torch.tensor(np.array(query_set), dtype=torch.float32),
        'query_labels': torch.tensor(query_labels, dtype=torch.long),
        'prototypes': torch.tensor(prototypes, dtype=torch.float32)
    }

# ==================== 评估函数 ====================
def evaluate(scores, query_labels):
    """评估指标计算"""
    preds = torch.argmax(scores, dim=-1)
    preds_np = preds.cpu().numpy()
    labels_np = query_labels.cpu().numpy()
    
    accuracy = accuracy_score(labels_np, preds_np)
    f1 = f1_score(labels_np, preds_np, average='macro')
    
    # mAP计算
    probs = F.softmax(scores, dim=-1).cpu().numpy()
    ap_scores = []
    for i in range(probs.shape[1]):
        y_true = (labels_np == i).astype(int)
        y_score = probs[:, i]
        if np.sum(y_true) > 0:
            ap = average_precision_score(y_true, y_score)
            ap_scores.append(ap)
    mAP = np.mean(ap_scores) if ap_scores else 0.0
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'mAP': mAP
    }

# ==================== 主实验函数 ====================
def run_experiment():
    """运行评分函数对比实验"""
    print("=== 小样本学习评分函数对比实验 ===")
    print("对比经典方法与近年创新方法")
    
    # 实验参数
    n_ways_list = [5, 10, 15]
    n_shots_list = [1, 3, 5]
    feature_dim = 512
    
    # 评分函数集合
    classic_funcs = {
        '余弦相似度': ClassicScoringFunctions.cosine_similarity,
        '欧式距离': ClassicScoringFunctions.euclidean_distance,
        '点积': ClassicScoringFunctions.dot_product
    }
    
    modern_funcs = {
        'ProtoNet距离': ModernScoringFunctions.prototypical_distance,
        'CosFace (2018)': ModernScoringFunctions.cosface_margin,
        'ArcFace (2019)': ModernScoringFunctions.arcface_margin,
        'NCA评分': ModernScoringFunctions.nca_score,
        'Soft NN (2019)': ModernScoringFunctions.soft_nn,
        'Triplet Margin': ModernScoringFunctions.triplet_margin
    }
    
    # 存储结果
    results = {**{name: {'accuracy': [], 'f1': [], 'mAP': []} for name in classic_funcs},
               **{name: {'accuracy': [], 'f1': [], 'mAP': []} for name in modern_funcs}}
    
    # Relation Network需要单独训练
    rn_results = {'accuracy': [], 'f1': [], 'mAP': []}
    
    # 运行实验
    for n_ways in n_ways_list:
        for n_shots in n_shots_list:
            print(f"\n--- 实验配置: {n_ways}类, {n_shots}样本/类 ---")
            
            # 生成数据
            data = generate_few_shot_data(
                n_ways=n_ways, 
                n_shots=n_shots, 
                n_queries=20,
                feature_dim=feature_dim
            )
            
            support = data['support'].to(device)
            query = data['query'].to(device)
            query_labels = data['query_labels'].to(device)
            
            # 计算每个类别的原型
            prototypes = []
            for way in range(n_ways):
                mask = data['support_labels'] == way
                proto = support[mask].mean(dim=0)
                prototypes.append(proto)
            prototypes = torch.stack(prototypes).to(device)
            
            # 经典评分函数
            for name, func in classic_funcs.items():
                scores = func(query, prototypes)
                metrics = evaluate(scores, query_labels)
                
                results[name]['accuracy'].append(metrics['accuracy'])
                results[name]['f1'].append(metrics['f1'])
                results[name]['mAP'].append(metrics['mAP'])
                
                print(f"  {name}: 准确率={metrics['accuracy']:.4f}")
            
            # 现代评分函数
            for name, func in modern_funcs.items():
                scores = func(query, prototypes)
                metrics = evaluate(scores, query_labels)
                
                results[name]['accuracy'].append(metrics['accuracy'])
                results[name]['f1'].append(metrics['f1'])
                results[name]['mAP'].append(metrics['mAP'])
                
                print(f"  {name}: 准确率={metrics['accuracy']:.4f}")
            
            # Relation Network（需要训练）
            rn = RelationNetwork(input_dim=feature_dim).to(device)
            optimizer = torch.optim.Adam(rn.parameters(), lr=0.001)
            
            # 简单训练
            for epoch in range(50):
                rn.train()
                optimizer.zero_grad()
                
                # 使用支持集训练
                support_proto = []
                for way in range(n_ways):
                    mask = data['support_labels'] == way
                    proto = support[mask].mean(dim=0)
                    support_proto.append(proto)
                support_proto = torch.stack(support_proto).to(device)
                
                # 计算关系分数
                rn_scores = rn(support, support_proto)
                
                # 使用交叉熵损失
                loss = F.cross_entropy(rn_scores, data['support_labels'].to(device))
                loss.backward()
                optimizer.step()
            
            # 评估
            rn.eval()
            with torch.no_grad():
                rn_scores = rn(query, prototypes)
                rn_metrics = evaluate(rn_scores, query_labels)
                
                rn_results['accuracy'].append(rn_metrics['accuracy'])
                rn_results['f1'].append(rn_metrics['f1'])
                rn_results['mAP'].append(rn_metrics['mAP'])
                
                print(f"  RelationNet (2017): 准确率={rn_metrics['accuracy']:.4f}")
    
    # 添加RelationNet结果
    results['RelationNet (2017)'] = rn_results
    
    return results, list(classic_funcs.keys()), list(modern_funcs.keys()) + ['RelationNet (2017)'], n_ways_list, n_shots_list

# ==================== 可视化函数 ====================
def visualize_results(results, classic_names, modern_names, n_ways_list, n_shots_list):
    """可视化实验结果"""
    all_names = classic_names + modern_names
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728', 
              '#7f7f7f', '#bcbd22', '#17becf', '#e377c2', '#8c564b']
    
    # 1. 经典方法vs现代方法对比（柱状图）
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, n_ways in enumerate(n_ways_list):
        ax = axes[idx]
        bar_width = 0.12
        x = np.arange(len(n_shots_list))
        
        for i, name in enumerate(all_names):
            start_idx = idx * len(n_shots_list)
            end_idx = start_idx + len(n_shots_list)
            acc_values = results[name]['accuracy'][start_idx:end_idx]
            bars = ax.bar(x + i * bar_width, acc_values, width=bar_width, label=name, color=colors[i])
            
            # 添加数值标注
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(f'{n_ways}类小样本分类', fontweight='bold')
        ax.set_xlabel('每类样本数')
        ax.set_ylabel('准确率')
        ax.set_xticks(x + bar_width * 4)
        ax.set_xticklabels([f'{s} shot' for s in n_shots_list])
        ax.set_ylim(0.5, 1.0)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classic_vs_modern.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 雷达图 - 综合性能对比
    labels = ['准确率', 'F1分数', 'mAP']
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    theta += theta[:1]
    
    for i, (name, color) in enumerate(zip(all_names, colors)):
        avg_acc = np.mean(results[name]['accuracy'])
        avg_f1 = np.mean(results[name]['f1'])
        avg_map = np.mean(results[name]['mAP'])
        
        values = [avg_acc, avg_f1, avg_map]
        values += values[:1]
        
        ax.plot(theta, values, linewidth=2, linestyle='-', label=name, color=color)
        ax.fill(theta, values, alpha=0.15, color=color)
    
    ax.set_thetagrids([t * 180 / np.pi for t in theta[:-1]], labels, fontsize=12)
    ax.set_title('评分函数综合性能对比', y=1.1, fontweight='bold', fontsize=16)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
    
    plt.savefig(os.path.join(output_dir, 'radar_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 样本数量敏感性分析
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (name, color) in enumerate(zip(all_names, colors)):
        all_values = np.array(results[name]['accuracy']).reshape(len(n_ways_list), len(n_shots_list))
        avg_values = all_values.mean(axis=0)
        ax.plot(n_shots_list, avg_values, marker='o', label=name, color=color, linewidth=2)
    
    ax.set_title('样本数量对不同评分函数的影响', fontweight='bold')
    ax.set_xlabel('每类样本数 (shots)')
    ax.set_ylabel('平均准确率')
    ax.legend(ncol=2)
    ax.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'sample_size_sensitivity.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 类别数量敏感性分析
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (name, color) in enumerate(zip(all_names, colors)):
        all_values = np.array(results[name]['accuracy']).reshape(len(n_ways_list), len(n_shots_list))
        avg_values = all_values.mean(axis=1)
        ax.plot(n_ways_list, avg_values, marker='s', label=name, color=color, linewidth=2)
    
    ax.set_title('类别数量对不同评分函数的影响', fontweight='bold')
    ax.set_xlabel('类别数量 (ways)')
    ax.set_ylabel('平均准确率')
    ax.legend(ncol=2)
    ax.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'ways_sensitivity.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

# ==================== 主函数 ====================
def main():
    # 运行实验
    results, classic_names, modern_names, n_ways_list, n_shots_list = run_experiment()
    
    # 可视化结果
    visualize_results(results, classic_names, modern_names, n_ways_list, n_shots_list)
    
    # 输出总结报告
    print("\n=== 实验总结报告 ===")
    print("=" * 60)
    
    all_names = classic_names + modern_names
    
    # 按平均准确率排序
    sorted_names = sorted(all_names, key=lambda x: np.mean(results[x]['accuracy']), reverse=True)
    
    print("\n【排名】按平均准确率排序:")
    for i, name in enumerate(sorted_names):
        avg_acc = np.mean(results[name]['accuracy'])
        avg_f1 = np.mean(results[name]['f1'])
        avg_map = np.mean(results[name]['mAP'])
        
        print(f"\n{i+1}. {name}:")
        print(f"   平均准确率: {avg_acc:.4f}")
        print(f"   平均F1分数: {avg_f1:.4f}")
        print(f"   平均mAP: {avg_map:.4f}")
    
    best_func = sorted_names[0]
    print(f"\n【结论】在小样本学习任务中，{best_func}表现最佳！")
    print(f"\n所有图表已保存至 {output_dir}/ 文件夹")

if __name__ == '__main__':
    main()