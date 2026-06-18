"""
项目配置文件
定义实验参数、数据路径等配置信息
"""

class Config:
    """项目配置类"""
    
    DATA_DIR = 'data'
    RESULTS_DIR = 'results'
    EXPERIMENTS_DIR = 'experiments'
    
    DATASETS = {
        'mnist': {
            'path': 'data/mnist/MNIST/raw',
            'num_classes': 10,
            'image_size': 28
        },
        'mnistm': {
            'path': 'data/mnistm',
            'num_classes': 10,
            'image_size': 28
        },
        'svhn': {
            'path': 'data/svhn',
            'num_classes': 10,
            'image_size': 32
        }
    }
    
    EXPERIMENT_SETTINGS = {
        'mnist_to_mnistm': {
            'source': 'mnist',
            'target': 'mnistm',
            'train_samples': 10000,
            'test_samples': 10000,
            'batch_size': 64,
            'baseline_epochs': 10,
            'finetune_pretrain_epochs': 10,
            'finetune_finetune_epochs': 5,
            'dann_epochs': 10,
            'learning_rate': 1e-3,
            'domain_loss_weight': 0.1
        }
    }
    
    DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    
    @classmethod
    def get_experiment_config(cls, experiment_name):
        """获取实验配置"""
        return cls.EXPERIMENT_SETTINGS.get(experiment_name, {})
    
    @classmethod
    def get_dataset_config(cls, dataset_name):
        """获取数据集配置"""
        return cls.DATASETS.get(dataset_name, {})

if __name__ == '__main__':
    config = Config()
    print("当前配置:")
    print(f"数据目录: {config.DATA_DIR}")
    print(f"结果目录: {config.RESULTS_DIR}")
    print(f"设备: {config.DEVICE}")
    print(f"\n可用数据集: {list(config.DATASETS.keys())}")
    print(f"可用实验: {list(config.EXPERIMENT_SETTINGS.keys())}")
