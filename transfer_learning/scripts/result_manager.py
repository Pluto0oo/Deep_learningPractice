"""
实验结果管理器 - 统一管理实验结果的保存和组织

功能包括：
1. 训练日志记录
2. 评估指标保存
3. 模型参数保存
4. 可视化结果保存
5. 实验报告生成
"""
import os
import json
import time
import torch
import numpy as np
from datetime import datetime

class ExperimentResultManager:
    """
    实验结果管理器 - 统一管理实验结果的保存和组织
    """
    
    def __init__(self, experiment_name, base_dir='results', report_dir='reports'):
        """
        初始化结果管理器
        
        Args:
            experiment_name: 实验名称
            base_dir: 结果保存的基础目录（模型、日志、指标等）
            report_dir: 报告保存目录（Markdown报告）
        """
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.report_dir = report_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(base_dir, f"{experiment_name}_{self.timestamp}")
        
        # 创建目录结构
        self._create_directories()
        
        # 初始化结果字典
        self.results = {
            'experiment_name': experiment_name,
            'timestamp': self.timestamp,
            'config': {},
            'training_log': [],
            'evaluation_metrics': {},
            'model_info': {},
            'system_info': {}
        }
        
        # 记录系统信息
        self._record_system_info()
    
    def _create_directories(self):
        """创建实验所需的目录结构"""
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'visualizations'), exist_ok=True)
        # 确保报告目录存在
        os.makedirs(self.report_dir, exist_ok=True)
    
    def _record_system_info(self):
        """记录系统信息"""
        import platform
        self.results['system_info'] = {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
        }
    
    def save_config(self, config):
        """保存实验配置"""
        self.results['config'] = config
        config_path = os.path.join(self.experiment_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    
    def log_training_epoch(self, epoch, loss, accuracy, **kwargs):
        """记录训练轮次信息"""
        log_entry = {
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy,
            'timestamp': time.time(),
            **kwargs
        }
        self.results['training_log'].append(log_entry)
        
        # 实时保存日志
        log_path = os.path.join(self.experiment_dir, 'logs', 'training.log')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"Epoch {epoch}: Loss={loss:.6f}, Acc={accuracy:.4f}\n")
    
    def save_evaluation_metrics(self, metrics, dataset_name='test'):
        """保存评估指标"""
        self.results['evaluation_metrics'][dataset_name] = metrics
        
        metrics_path = os.path.join(self.experiment_dir, 'metrics', f'{dataset_name}.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
    
    def save_model(self, model, model_name='model', epoch=None):
        """保存模型参数"""
        if epoch is not None:
            model_path = os.path.join(self.experiment_dir, 'models', f'{model_name}_epoch_{epoch}.pth')
        else:
            model_path = os.path.join(self.experiment_dir, 'models', f'{model_name}.pth')
        
        torch.save(model.state_dict(), model_path)
        
        # 记录模型信息
        self.results['model_info'][model_name] = {
            'path': model_path,
            'params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    
    def save_metrics_plot(self, metric_name, values, title=None):
        """保存指标可视化图"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 5))
            plt.plot(values)
            plt.title(title if title else f'{metric_name} over epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.grid(True)
            
            plot_path = os.path.join(self.experiment_dir, 'visualizations', f'{metric_name}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return plot_path
        except ImportError:
            print("Warning: matplotlib not installed, skipping plot generation")
            return None
    
    def save_confusion_matrix(self, y_true, y_pred, class_names=None, title='Confusion Matrix'):
        """保存混淆矩阵可视化"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(title)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            plot_path = os.path.join(self.experiment_dir, 'visualizations', 'confusion_matrix.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return plot_path
        except ImportError:
            print("Warning: matplotlib/seaborn not installed, skipping confusion matrix")
            return None
    
    def generate_report(self):
        """生成实验报告"""
        report = f"""# {self.experiment_name} - 实验报告

## 一、实验概述

**实验名称**: {self.experiment_name}  
**实验时间**: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}

---

## 二、实验配置

```json
{json.dumps(self.results['config'], indent=2, ensure_ascii=False)}
```

---

## 三、系统信息

| 项目 | 值 |
|------|------|
| 操作系统 | {self.results['system_info']['platform']} |
| Python版本 | {self.results['system_info']['python_version']} |
| CUDA可用 | {self.results['system_info']['cuda_available']} |
| CUDA设备 | {self.results['system_info']['cuda_device']} |

---

## 四、训练日志

### 4.1 训练曲线摘要

| 轮次 | 损失 | 准确率 |
|------|------|--------|
"""
        
        # 添加训练日志表格
        for log in self.results['training_log']:
            report += f"| {log['epoch']} | {log['loss']:.6f} | {log['accuracy']:.4f} |\n"
        
        # 添加评估指标
        report += """
---

## 五、评估指标

"""
        for dataset_name, metrics in self.results['evaluation_metrics'].items():
            report += f"### 5.1 {dataset_name}数据集\n\n"
            report += "| 指标 | 值 |\n"
            report += "|------|------|\n"
            for key, value in metrics.items():
                report += f"| {key} | {value} |\n"
        
        # 添加模型信息
        report += """
---

## 六、模型信息

"""
        for model_name, info in self.results['model_info'].items():
            report += f"### 6.1 {model_name}\n\n"
            report += f"- **保存路径**: {info['path']}\n"
            report += f"- **参数数量**: {info['params']:,}\n"
            report += f"- **可训练参数**: {info['trainable_params']:,}\n"
        
        report += f"""
---

## 七、实验结果文件

| 类型 | 路径 |
|------|------|
| 配置文件 | `{os.path.join(self.experiment_dir, 'config.json')}` |
| 训练日志 | `{os.path.join(self.experiment_dir, 'logs', 'training.log')}` |
| 评估指标 | `{os.path.join(self.experiment_dir, 'metrics')}` |
| 模型文件 | `{os.path.join(self.experiment_dir, 'models')}` |
| 可视化结果 | `{os.path.join(self.experiment_dir, 'visualizations')}` |

---

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 将报告保存到reports目录
        report_path = os.path.join(self.report_dir, f"{self.experiment_name}_{self.timestamp}_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 同时在实验目录保存一份副本
        local_report_path = os.path.join(self.experiment_dir, 'experiment_report.md')
        with open(local_report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report_path
    
    def save_summary(self):
        """保存结果摘要到JSON文件"""
        summary_path = os.path.join(self.experiment_dir, 'results_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
        
        return summary_path
    
    def get_experiment_dir(self):
        """获取实验目录路径"""
        return self.experiment_dir

    def __str__(self):
        return f"ExperimentResultManager({self.experiment_name}, {self.experiment_dir})"


# 便捷函数
def create_experiment(experiment_name, config=None):
    """
    创建实验结果管理器的便捷函数
    
    Args:
        experiment_name: 实验名称
        config: 实验配置字典
    
    Returns:
        ExperimentResultManager实例
    """
    manager = ExperimentResultManager(experiment_name)
    if config is not None:
        manager.save_config(config)
    return manager


def load_experiment(experiment_dir):
    """
    加载已保存的实验结果
    
    Args:
        experiment_dir: 实验目录路径
    
    Returns:
        ExperimentResultManager实例（包含已加载的结果）
    """
    summary_path = os.path.join(experiment_dir, 'results_summary.json')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Experiment summary not found at {summary_path}")
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 从路径中提取实验名称
    experiment_name = os.path.basename(experiment_dir).split('_')[0]
    
    manager = ExperimentResultManager(experiment_name)
    manager.results = results
    manager.experiment_dir = experiment_dir
    
    return manager


if __name__ == '__main__':
    # 示例用法
    print("测试ExperimentResultManager...")
    
    # 创建实验管理器
    manager = create_experiment('test_experiment', {
        'model': 'SimpleCNN',
        'dataset': 'MNIST',
        'epochs': 10,
        'batch_size': 64,
        'lr': 0.001
    })
    
    # 模拟训练日志
    for epoch in range(1, 6):
        manager.log_training_epoch(
            epoch=epoch,
            loss=1.0 / epoch,
            accuracy=0.5 + epoch * 0.1,
            learning_rate=0.001 * (0.9 ** epoch)
        )
    
    # 模拟评估指标
    manager.save_evaluation_metrics({
        'accuracy': 0.85,
        'precision': 0.84,
        'recall': 0.83,
        'f1_score': 0.835
    }, 'test')
    
    # 模拟模型保存
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 2)
    
    model = SimpleModel()
    manager.save_model(model, 'simple_model')
    
    # 生成报告
    report_path = manager.generate_report()
    print(f"报告已生成: {report_path}")
    
    # 保存摘要
    summary_path = manager.save_summary()
    print(f"摘要已保存: {summary_path}")
    
    print(f"\n实验目录: {manager.get_experiment_dir()}")
    print("测试完成！")