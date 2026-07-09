import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    return train_transform, test_transform


def load_dataset(data_dir: str = "./data", dataset_name: str = "svhn") -> Tuple:
    train_transform, test_transform = get_transforms()

    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform
        )

        test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=test_transform
        )
    elif dataset_name == "mnist":
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        )
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        )
    elif dataset_name == "svhn":
        svhn_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        svhn_train_path = os.path.join(data_dir, 'SVHN', 'train_32x32.mat')
        svhn_test_path = os.path.join(data_dir, 'SVHN', 'test_32x32.mat')
        
        train_download = not os.path.exists(svhn_train_path)
        test_download = not os.path.exists(svhn_test_path)
        
        if train_download:
            print("SVHN train set not found locally, downloading...")
        else:
            print("Using local SVHN train set...")
        if test_download:
            print("SVHN test set not found locally, downloading...")
        else:
            print("Using local SVHN test set...")
            
        train_dataset = datasets.SVHN(
            root=data_dir,
            split='train',
            download=train_download,
            transform=svhn_transform
        )
        test_dataset = datasets.SVHN(
            root=data_dir,
            split='test',
            download=test_download,
            transform=svhn_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_dataset, test_dataset


def partition_data_dirichlet(
    dataset,
    num_clients: int,
    alpha: float = 0.5
) -> Dict[int, List[int]]:
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    client_indices = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()

        cumulative = np.cumsum(proportions)
        cumulative[-1] = 1.0

        split_points = (cumulative * len(class_indices)).astype(int)
        prev = 0

        for i in range(num_clients):
            client_indices[i].extend(class_indices[prev:split_points[i]].tolist())
            prev = split_points[i]

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return client_indices


def partition_data_iid(
    dataset,
    num_clients: int
) -> Dict[int, List[int]]:
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split_size = len(indices) // num_clients

    client_indices = {}
    for i in range(num_clients):
        start = i * split_size
        end = start + split_size if i < num_clients - 1 else len(indices)
        client_indices[i] = indices[start:end]

    return client_indices


def get_client_dataloader(
    dataset,
    client_indices: List[int],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
):
    subset = Subset(dataset, client_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def prepare_client_data(
    config: Dict
) -> Tuple[Dict[int, DataLoader], DataLoader, Dict[int, int]]:
    data_dir = config['data'].get('data_dir', './data')
    num_clients = config['experiment'].get('num_clients', 3)
    partition_type = config['data'].get('partition_type', 'iid')
    batch_size = config['data'].get('batch_size', 32)
    num_workers = config['data'].get('num_workers', 0)
    dataset_name = config['data'].get('dataset', 'svhn')

    train_dataset, test_dataset = load_dataset(data_dir, dataset_name)

    if partition_type == 'iid':
        client_indices = partition_data_iid(train_dataset, num_clients)
    else:
        alpha = config['data'].get('dirichlet_alpha', 0.5)
        client_indices = partition_data_dirichlet(train_dataset, num_clients, alpha)

    client_loaders = {}
    client_sizes = {}

    for client_id in range(num_clients):
        client_loaders[client_id] = get_client_dataloader(
            train_dataset,
            client_indices[client_id],
            batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        client_sizes[client_id] = len(client_indices[client_id])

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return client_loaders, test_loader, client_sizes