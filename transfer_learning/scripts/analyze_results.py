import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

def load_results(results_path):
    """加载实验结果"""
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_results(results, save_dir):
    """绘制实验结果对比图"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    for exp_name, exp_results in results.items():
        methods = ['baseline', 'dann', 'finetune_pretrained', 'finetune_scratch']
        method_names = ['基线', 'DANN', '微调(预训练)', '微调(从头)']
        means = [exp_results[m]['mean'] for m in methods]
        stds = [exp_results[m]['std'] for m in methods]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(methods))
        width = 0.6
        
        bars = ax.bar(x, means, width, yerr=stds, capsize=5, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        ax.set_xlabel('方法', fontsize=12)
        ax.set_ylabel('准确率', fontsize=12)
        ax.set_title(f'{exp_name} - 迁移学习方法对比', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(method_names)
        ax.set_ylim(0, 1)
        
        # 在柱状图上显示数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{exp_name.replace(" -> ", "_")}_comparison.png'), dpi=300)
        plt.close()

def generate_report(results):
    """生成实验报告"""
    report = []
    report.append("# 迁移学习对比实验报告")
    report.append("")
    report.append("## 实验概述")
    report.append("本实验对比了不同迁移学习方法在 MNIST -> MNIST-M 任务上的性能表现。")
    report.append("")
    report.append("## 实验设置")
    report.append("- **源域**: MNIST (手写数字)")
    report.append("- **目标域**: MNIST-M (混合背景手写数字)")
    report.append("- **评价指标**: 分类准确率")
    report.append("- **重复次数**: 3次")
    report.append("")
    report.append("## 实验结果")
    report.append("")
    
    for exp_name, exp_results in results.items():
        report.append(f"### {exp_name}")
        report.append("")
        report.append("| 方法 | 准确率(均值) | 标准差 |")
        report.append("|------|-------------|--------|")
        
        methods = [('baseline', '基线'), 
                   ('dann', 'DANN'), 
                   ('finetune_pretrained', '微调(预训练)'), 
                   ('finetune_scratch', '微调(从头)')]
        
        for method_key, method_name in methods:
            mean = exp_results[method_key]['mean']
            std = exp_results[method_key]['std']
            report.append(f"| {method_name} | {mean:.4f} | {std:.4f} |")
        
        report.append("")
    
    report.append("## 结果分析")
    report.append("")
    report.append("1. **DANN方法**: 通过域对抗训练，模型能够学习到域不变的特征表示，")
    report.append("   在迁移学习任务上表现出色。")
    report.append("")
    report.append("2. **微调方法**: 预训练模型在迁移学习任务上通常比从头训练效果更好，")
    report.append("   这体现了预训练的价值。")
    report.append("")
    report.append("3. **基线方法**: 直接在目标域训练的效果通常较差，")
    report.append("   特别是当目标域数据有限时。")
    report.append("")
    report.append("## 结论")
    report.append("迁移学习方法能够有效提升模型在不同域之间的泛化能力，")
    report.append("其中DANN和预训练微调是两种有效的迁移学习策略。")
    
    return '\n'.join(report)

def main():
    """分析实验结果"""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    results_path = os.path.join(results_dir, 'experiment_results.json')
    
    if not os.path.exists(results_path):
        print(f"错误: 结果文件不存在 - {results_path}")
        print("请先运行 run_experiments.py 生成实验结果")
        return
    
    # 加载结果
    results = load_results(results_path)
    
    # 绘制结果
    plot_results(results, results_dir)
    print("结果图已生成")
    
    # 生成报告
    report = generate_report(results)
    report_path = os.path.join(results_dir, 'experiment_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"报告已保存到: {report_path}")
    
    # 打印汇总
    print("\n" + "="*60)
    print("实验结果汇总")
    print("="*60)
    
    for exp_name, exp_results in results.items():
        print(f"\n{exp_name}:")
        methods = [('baseline', '基线'), 
                   ('dann', 'DANN'), 
                   ('finetune_pretrained', '微调(预训练)'), 
                   ('finetune_scratch', '微调(从头)')]
        
        for method_key, method_name in methods:
            mean = exp_results[method_key]['mean']
            std = exp_results[method_key]['std']
            print(f"  {method_name}: {mean:.4f} ± {std:.4f}")

if __name__ == '__main__':
    main()