import os
import pickle
import numpy as np
import scipy.io as sio
import torchvision
import torchvision.transforms as transforms

class MNISTLoader:
    def __init__(self, data_dir):
        self.data_dir = os.path.join(data_dir, 'mnist')
    
    def load(self):
        transform = transforms.Compose([transforms.ToTensor()])
        
        train_set = torchvision.datasets.MNIST(root=self.data_dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=self.data_dir, train=False, download=True, transform=transform)
        
        # 转换为numpy数组
        train_images = []
        train_labels = []
        for img, label in train_set:
            train_images.append(img.numpy())
            train_labels.append(label)
        
        test_images = []
        test_labels = []
        for img, label in test_set:
            test_images.append(img.numpy())
            test_labels.append(label)
        
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)
        
        # 转换为RGB (1通道 -> 3通道)
        train_images = np.repeat(train_images, 3, axis=1)
        test_images = np.repeat(test_images, 3, axis=1)
        
        # 转换为 (N, H, W, C) 格式
        train_images = np.transpose(train_images, (0, 2, 3, 1))
        test_images = np.transpose(test_images, (0, 2, 3, 1))
        
        return (train_images, train_labels), (test_images, test_labels)

class MNISTMLoader:
    def __init__(self, data_dir):
        self.data_dir = os.path.join(data_dir, 'mnistm')
    
    def load(self):
        pkl_path = os.path.join(self.data_dir, 'mnistm_data.pkl')
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        train_images = data['train']
        train_labels = data['train_labels']
        test_images = data['test']
        test_labels = data['test_labels']
        
        return (train_images.astype(np.float32) / 255.0, train_labels), \
               (test_images.astype(np.float32) / 255.0, test_labels)

class SVHNLoader:
    def __init__(self, data_dir):
        self.data_dir = os.path.join(data_dir, 'svhn')
    
    def load(self):
        train_path = os.path.join(self.data_dir, 'train_32x32.mat')
        test_path = os.path.join(self.data_dir, 'test_32x32.mat')
        
        train_data = sio.loadmat(train_path)
        test_data = sio.loadmat(test_path)
        
        train_images = np.transpose(train_data['X'], (3, 0, 1, 2)).astype(np.float32) / 255.0
        train_labels = train_data['y'].flatten() % 10
        test_images = np.transpose(test_data['X'], (3, 0, 1, 2)).astype(np.float32) / 255.0
        test_labels = test_data['y'].flatten() % 10
        
        return (train_images, train_labels), (test_images, test_labels)

class DataLoader:
    def __init__(self, data_dir='data'):
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(os.path.dirname(__file__), '..', data_dir)
        self.data_dir = data_dir
    
    def get_mnist(self):
        loader = MNISTLoader(self.data_dir)
        return loader.load()
    
    def get_mnistm(self):
        loader = MNISTMLoader(self.data_dir)
        return loader.load()
    
    def get_svhn(self):
        loader = SVHNLoader(self.data_dir)
        return loader.load()
    
    def get_domain_dataset(self, source='mnist', target='mnistm'):
        """获取源域和目标域数据集"""
        if source == 'mnist':
            (src_train_imgs, src_train_labels), (src_test_imgs, src_test_labels) = self.get_mnist()
        elif source == 'svhn':
            (src_train_imgs, src_train_labels), (src_test_imgs, src_test_labels) = self.get_svhn()
        else:
            raise ValueError(f"Unknown source domain: {source}")
        
        if target == 'mnistm':
            (tgt_train_imgs, tgt_train_labels), (tgt_test_imgs, tgt_test_labels) = self.get_mnistm()
        elif target == 'mnist':
            (tgt_train_imgs, tgt_train_labels), (tgt_test_imgs, tgt_test_labels) = self.get_mnist()
        elif target == 'svhn':
            (tgt_train_imgs, tgt_train_labels), (tgt_test_imgs, tgt_test_labels) = self.get_svhn()
        else:
            raise ValueError(f"Unknown target domain: {target}")
        
        return {
            'source': {
                'train': (src_train_imgs, src_train_labels),
                'test': (src_test_imgs, src_test_labels)
            },
            'target': {
                'train': (tgt_train_imgs, tgt_train_labels),
                'test': (tgt_test_imgs, tgt_test_labels)
            }
        }

def main():
    """测试数据加载器"""
    loader = DataLoader()
    
    print("=== 测试数据加载器 ===")
    
    try:
        (train_imgs, train_labels), (test_imgs, test_labels) = loader.get_mnist()
        print(f"MNIST - Train: {train_imgs.shape}, Labels: {train_labels.shape}")
        print(f"MNIST - Test: {test_imgs.shape}, Labels: {test_labels.shape}")
    except Exception as e:
        print(f"MNIST加载失败: {e}")
    
    try:
        (train_imgs, train_labels), (test_imgs, test_labels) = loader.get_mnistm()
        print(f"MNIST-M - Train: {train_imgs.shape}, Labels: {train_labels.shape}")
        print(f"MNIST-M - Test: {test_imgs.shape}, Labels: {test_labels.shape}")
    except Exception as e:
        print(f"MNIST-M加载失败: {e}")
    
    try:
        (train_imgs, train_labels), (test_imgs, test_labels) = loader.get_svhn()
        print(f"SVHN - Train: {train_imgs.shape}, Labels: {train_labels.shape}")
        print(f"SVHN - Test: {test_imgs.shape}, Labels: {test_labels.shape}")
    except Exception as e:
        print(f"SVHN加载失败: {e}")

if __name__ == '__main__':
    main()