import os
import sys
import json
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.data_loader import DataLoader
from models.dann import DANN, train_dann, evaluate_dann, get_loader

def run_dann_experiment(source_domain, target_domain, device, num_epochs=10, sample_size=10000):
    print(f"\n=== DANN实验: {source_domain} -> {target_domain} ===")
    
    data_loader = DataLoader()
    dataset = data_loader.get_domain_dataset(source=source_domain, target=target_domain)
    
    src_train_imgs, src_train_labels = dataset['source']['train']
    tgt_train_imgs, _ = dataset['target']['train']
    tgt_test_imgs, tgt_test_labels = dataset['target']['test']
    
    # 使用部分数据进行训练以控制时间
    src_train_imgs = src_train_imgs[:sample_size]
    src_train_labels = src_train_labels[:sample_size]
    tgt_train_imgs = tgt_train_imgs[:sample_size]
    
    print(f"源域训练数据: {src_train_imgs.shape}")
    print(f"目标域训练数据: {tgt_train_imgs.shape}")
    print(f"目标域测试数据: {tgt_test_imgs.shape}")
    
    source_loader = get_loader(src_train_imgs, src_train_labels, batch_size=64)
    target_loader = get_loader(tgt_train_imgs, np.zeros(len(tgt_train_imgs)), batch_size=64)
    test_loader = get_loader(tgt_test_imgs, tgt_test_labels, batch_size=64, shuffle=False)
    
    model = DANN(num_classes=10)
    model = train_dann(model, source_loader, target_loader, device, num_epochs=num_epochs)
    
    accuracy = evaluate_dann(model, test_loader, device)
    print(f"DANN准确率: {accuracy:.4f}")
    
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    print("\n=== 迁移学习对比实验 ===")
    print("实验配置: MNIST -> MNIST-M")
    print("训练轮数: 10")
    print("每轮数据量: 10000")
    
    # 运行多次实验取平均
    runs = 2
    accuracies = []
    
    for i in range(runs):
        print(f"\n--- 运行 {i+1}/{runs} ---")
        acc = run_dann_experiment('mnist', 'mnistm', device, num_epochs=10, sample_size=10000)
        accuracies.append(acc)
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\n=== 实验结果汇总 ===")
    print(f"平均准确率: {mean_acc:.4f}")
    print(f"标准差: {std_acc:.4f}")
    
    results = {
        'MNIST -> MNIST-M': {
            'dann': {
                'mean': float(mean_acc),
                'std': float(std_acc),
                'runs': [float(a) for a in accuracies],
                'config': {
                    'epochs': 10,
                    'sample_size': 10000,
                    'device': str(device)
                }
            }
        }
    }
    
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, 'experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n结果已保存到: {results_path}")

if __name__ == '__main__':
    main()