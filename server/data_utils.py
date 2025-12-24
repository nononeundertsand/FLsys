import torch
from torchvision import datasets, transforms
import numpy as np
import os

def get_dataset(name):
    """下载并获取原始数据集"""
    data_dir = './data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if name == 'FashionMNIST':
        # --- [修复核心] 强制替换下载源为 GitHub ---
        # 原始 AWS 源经常 502 挂掉，这里改为 GitHub 镜像
        datasets.FashionMNIST.mirrors = [
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",  # 原始源(备用)
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/" # GitHub源(优先)
        ]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        print("[*] 正在下载 FashionMNIST (如长时间未动请检查网络)...")
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"暂不支持数据集: {name}")
    
    return train_dataset, test_dataset

def dirichlet_partition(train_dataset, n_clients, alpha=0.5):
    """
    根据迪利克雷分布划分数据
    """
    n_classes = len(train_dataset.classes)
    labels = np.array(train_dataset.targets)
    client_indices = [[] for _ in range(n_clients)]
    
    for k in range(n_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        split_idx = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        idx_splits = np.split(idx_k, split_idx)
        for i in range(n_clients):
            client_indices[i].extend(idx_splits[i])

    total_samples = sum([len(c) for c in client_indices])
    print(f"[数据划分] 原始样本: {len(labels)}, 分配样本: {total_samples}")
    
    return client_indices