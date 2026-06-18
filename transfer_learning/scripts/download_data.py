import os
import sys
import numpy as np
import pickle
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def download_mnist_with_torch():
    """使用torchvision下载MNIST数据集"""
    try:
        import torchvision
        import torchvision.transforms as transforms
        
        mnist_dir = os.path.join(DATA_DIR, 'mnist')
        os.makedirs(mnist_dir, exist_ok=True)
        
        transform = transforms.Compose([transforms.ToTensor()])
        
        train_set = torchvision.datasets.MNIST(root=mnist_dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=mnist_dir, train=False, download=True, transform=transform)
        
        print("MNIST数据集下载成功")
        return True
    except Exception as e:
        print(f"使用torchvision下载MNIST失败: {e}")
        return False

def download_svhn():
    """下载SVHN数据集"""
    try:
        import scipy.io as sio
        import requests
        from tqdm import tqdm
        
        svhn_dir = os.path.join(DATA_DIR, 'svhn')
        os.makedirs(svhn_dir, exist_ok=True)
        
        urls = [
            'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
            'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
        ]
        
        for url in urls:
            filename = os.path.basename(url)
            filepath = os.path.join(svhn_dir, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                with open(filepath, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                        for data in response.iter_content(1024):
                            f.write(data)
                            pbar.update(len(data))
        
        print("SVHN数据集下载成功")
        return True
    except Exception as e:
        print(f"下载SVHN失败: {e}")
        return False

def generate_mnistm_like_data():
    """生成类似MNIST-M的数据"""
    mnistm_dir = os.path.join(DATA_DIR, 'mnistm')
    os.makedirs(mnistm_dir, exist_ok=True)
    
    save_path = os.path.join(mnistm_dir, 'mnistm_data.pkl')
    if os.path.exists(save_path):
        print("MNIST-M数据已存在")
        return True
    
    try:
        import torchvision
        import torchvision.transforms as transforms
        
        # 加载MNIST
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.MNIST(root=os.path.join(DATA_DIR, 'mnist'), 
                                               train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=os.path.join(DATA_DIR, 'mnist'), 
                                              train=False, download=True, transform=transform)
        
        # 转换为numpy
        train_images = []
        train_labels = []
        for img, label in train_set:
            train_images.append(img.numpy().squeeze() * 255)
            train_labels.append(label)
        
        test_images = []
        test_labels = []
        for img, label in test_set:
            test_images.append(img.numpy().squeeze() * 255)
            test_labels.append(label)
        
        train_images = np.array(train_images, dtype=np.uint8)
        train_labels = np.array(train_labels, dtype=np.int64)
        test_images = np.array(test_images, dtype=np.uint8)
        test_labels = np.array(test_labels, dtype=np.int64)
        
        # 创建MNIST-M风格数据（添加随机背景）
        def add_background(images):
            result = np.zeros((len(images), 28, 28, 3), dtype=np.uint8)
            np.random.seed(42)
            for i, img in enumerate(images):
                bg_color = np.random.randint(0, 256, size=3)
                for c in range(3):
                    result[i, :, :, c] = bg_color[c] * (1 - img/255) + img * (1 if c == 0 else 0)
            return result
        
        print("生成MNIST-M风格训练数据...")
        train_mnistm = add_background(train_images)
        print("生成MNIST-M风格测试数据...")
        test_mnistm = add_background(test_images)
        
        mnistm_data = {
            'train': train_mnistm,
            'train_labels': train_labels,
            'test': test_mnistm,
            'test_labels': test_labels
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(mnistm_data, f)
        print("MNIST-M风格数据生成成功")
        return True
    except Exception as e:
        print(f"生成MNIST-M失败: {e}")
        return False

def main():
    print("=== 迁移学习数据集下载脚本 ===")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 使用torchvision下载MNIST
    download_mnist_with_torch()
    
    # 下载SVHN
    download_svhn()
    
    # 生成MNIST-M风格数据
    generate_mnistm_like_data()
    
    print("\n=== 所有数据集下载完成 ===")
    print(f"数据存储位置: {DATA_DIR}")

if __name__ == '__main__':
    main()