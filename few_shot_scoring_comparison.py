import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split

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

# ==================== 评分函数定义 ====================
class ScoringFunctions:
    """不同评分函数实现"""
    
    @staticmethod
    def cosine_similarity(x1, x2):
        """余弦相似度：越接近1越相似"""
        x1_norm = F.normalize(x1, dim=-1)
        x2_norm = F.normalize(x2, dim=-1)
        return torch.matmul(x1_norm, x2_norm.transpose(-2, -1))
    
    @staticmethod
    def euclidean_distance(x1, x2):
        """欧式距离：越接近0越相似，取负值使其与相似度方向一致"""
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        dist = torch.norm(diff, dim=-1)
        return -dist  # 取负，使越大越相似
    
    @staticmethod
    def dot_product(x1, x2):
        """点积：越大越相似"""
        return torch.matmul(x1, x2.transpose(-2, -1))
    
    @staticmethod
    def manhattan_distance(x1, x2):
        """曼哈顿距离：越接近0越相似，取负值"""
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        dist = torch.abs(diff).sum(dim=-1)
        return -dist  # 取负，使越大越相似
    
    @staticmethod
    def cosine_with_temperature(x1, x2, temperature=0.1):
        """带温度系数的余弦相似度"""
        x1_norm = F.normalize(x1, dim=-1)
        x2_norm = F.normalize(x2, dim=-1)
        return torch.matmul(x1_norm, x2_norm.transpose(-2, -1)) / temperature

# ==================== 小样本学习模型 ====================
class FewShotLearner(nn.Module):
    """简单的小样本学习模型"""
    
    def __init__(self, input_dim=512, hidden_dim=256):
        super(FewShotLearner, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ==================== 实验设置 ====================
def generate_few_shot_data(n_ways=5, n_shots=5, n_queries=15, feature_dim=512, n_samples=1000):
    """生成小样本学习数据集"""
    np.random.seed(42)
    
    # 为每个类别生成一个原型特征
    prototypes = np.random.randn(n_ways, feature_dim)
    
    # 生成支持集和查询集
    support_set = []
    support_labels = []
    query_set = []
    query_labels = []
    
    for way in range(n_ways):
        # 生成支持样本（围绕原型添加噪声）
        for _ in range(n_shots):
            sample = prototypes[way] + np.random.randn(feature_dim) * 0.1
            support_set.append(sample)
            support_labels.append(way)
        
        # 生成查询样本
        for _ in range(n_queries):
            sample = prototypes[way] + np.random.randn(feature_dim) * 0.15
            query_set.append(sample)
            query_labels.append(way)
    
    return {
        'support': torch.tensor(support_set, dtype=torch.float32),
        'support_labels': torch.tensor(support_labels, dtype=torch.long),
        'query': torch.tensor(query_set, dtype=torch.float32),
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
    
    # mAP计算（简化版）
    probs = F.softmax(scores, dim=-1).cpu().numpy()
    ap_scores = []
    for i in range(probs.shape[1]):  # 每个类别
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
    print("=== 开始小样本评分函数对比实验 ===")
    
    # 实验参数
    n_ways_list = [5, 10, 15]  # 类别数
    n_shots_list = [1, 3, 5, 10]  # 每个类别的样本数
    feature_dim = 512
    
    scoring_functions = {
        '余弦相似度': ScoringFunctions.cosine_similarity,
        '欧式距离': ScoringFunctions.euclidean_distance,
        '点积': ScoringFunctions.dot_product,
        '曼哈顿距离': ScoringFunctions.manhattan_distance,
        '带温度余弦': lambda x, y: ScoringFunctions.cosine_with_temperature(x, y, temperature=0.1)
    }
    
    # 存储结果
    results = {name: {'accuracy': [], 'f1': [], 'mAP': []} for name in scoring_functions}
    
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
            
            # 计算每个类别的原型（支持集均值）
            prototypes = []
            for way in range(n_ways):
                mask = data['support_labels'] == way
                proto = support[mask].mean(dim=0)
                prototypes.append(proto)
            prototypes = torch.stack(prototypes).to(device)
            
            # 对每个评分函数进行评估
            for name, func in scoring_functions.items():
                scores = func(query, prototypes)
                metrics = evaluate(scores, query_labels)
                
                results[name]['accuracy'].append(metrics['accuracy'])
                results[name]['f1'].append(metrics['f1'])
                results[name]['mAP'].append(metrics['mAP'])
                
                print(f"  {name}: 准确率={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, mAP={metrics['mAP']:.4f}")
    
    return results, scoring_functions.keys(), n_ways_list, n_shots_list

# ==================== 可视化函数 ====================
def visualize_results(results, function_names, n_ways_list, n_shots_list):
    """可视化实验结果"""
    # 1. 不同评分函数的准确率对比（柱状图）
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, n_ways in enumerate(n_ways_list):
        ax = axes[idx]
        bar_width = 0.15
        x = np.arange(len(n_shots_list))
        
        for i, name in enumerate(function_names):
            start_idx = idx * len(n_shots_list)
            end_idx = start_idx + len(n_shots_list)
            acc_values = results[name]['accuracy'][start_idx:end_idx]
            ax.bar(x + i * bar_width, acc_values, width=bar_width, label=name)
        
        ax.set_title(f'{n_ways} 类别的评分函数对比', fontweight='bold')
        ax.set_xlabel('每类样本数 (shots)')
        ax.set_ylabel('准确率')
        ax.set_xticks(x + bar_width * 2)
        ax.set_xticklabels([f'{s} shot' for s in n_shots_list])
        ax.legend()
        ax.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 综合指标雷达图
    labels = ['准确率', 'F1分数', 'mAP']
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    theta += theta[:1]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
    
    for i, (name, color) in enumerate(zip(function_names, colors)):
        # 取所有实验的平均指标
        avg_acc = np.mean(results[name]['accuracy'])
        avg_f1 = np.mean(results[name]['f1'])
        avg_map = np.mean(results[name]['mAP'])
        
        values = [avg_acc, avg_f1, avg_map]
        values += values[:1]
        
        ax.plot(theta, values, linewidth=2, linestyle='-', label=name, color=color)
        ax.fill(theta, values, alpha=0.2, color=color)
    
    ax.set_thetagrids([t * 180 / np.pi for t in theta[:-1]], labels)
    ax.set_title('评分函数综合性能对比', y=1.1, fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.savefig(os.path.join(output_dir, 'radar_chart.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 样本数量对性能的影响（折线图）
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['accuracy', 'f1', 'mAP']
    metric_names = ['准确率', 'F1分数', 'mAP']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        
        for i, (name, color) in enumerate(zip(function_names, colors)):
            # 按n_ways分组，取平均值
            all_values = np.array(results[name][metric]).reshape(len(n_ways_list), len(n_shots_list))
            avg_values = all_values.mean(axis=0)  # 不同n_ways的平均
            
            ax.plot(n_shots_list, avg_values, marker='o', label=name, color=color)
        
        ax.set_title(f'{metric_name} vs 每类样本数', fontweight='bold')
        ax.set_xlabel('每类样本数')
        ax.set_ylabel(metric_name)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_size_effect.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 类别数量对性能的影响
    fig, plt.subplots(figsize=(10, 6))
    
    for i, (name, color) in enumerate(zip(function_names, colors)):
        all_values = np.array(results[name]['accuracy']).reshape(len(n_ways_list), len(n_shots_list))
        avg_values = all_values.mean(axis=1)  # 不同shots的平均
        
        plt.plot(n_ways_list, avg_values, marker='s', label=name, color=color, linewidth=2)
    
    plt.title('准确率 vs 类别数量', fontweight='bold')
    plt.xlabel('类别数量 (ways)')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'num_ways_effect.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

# ==================== 主函数 ====================
def main():
    # 运行实验
    results, function_names, n_ways_list, n_shots_list = run_experiment()
    
    # 可视化结果
    visualize_results(results, function_names, n_ways_list, n_shots_list)
    
    # 输出总结报告
    print("\n=== 实验总结报告 ===")
    print("=" * 50)
    
    for name in function_names:
        avg_acc = np.mean(results[name]['accuracy'])
        avg_f1 = np.mean(results[name]['f1'])
        avg_map = np.mean(results[name]['mAP'])
        
        print(f"\n{name}:")
        print(f"  平均准确率: {avg_acc:.4f}")
        print(f"  平均F1分数: {avg_f1:.4f}")
        print(f"  平均mAP: {avg_map:.4f}")
    
    # 找出最佳评分函数
    best_func = max(function_names, key=lambda x: np.mean(results[x]['accuracy']))
    print(f"\n【结论】在小样本学习任务中，{best_func}表现最佳！")
    
    print(f"\n所有图表已保存至 {output_dir}/ 文件夹")

if __name__ == '__main__':
    main()