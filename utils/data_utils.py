import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os
import numpy as np
from PIL import Image

class MNISTDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.dataset = datasets.MNIST(
            root=root_dir,
            train=train,
            download=True,
            transform=transform or transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class EMNISTDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.dataset = datasets.EMNIST(
            root=root_dir,
            split='letters',
            train=train,
            download=True,
            transform=transform or transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class QuickDrawDataset(Dataset):
    def __init__(self, root_dir, category, transform=None):
        self.root_dir = os.path.join(root_dir, category)
        self.transform = transform or transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.file_list = [f for f in os.listdir(self.root_dir) if f.endswith('.npy')]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        data = np.load(os.path.join(self.root_dir, self.file_list[idx]))
        image = Image.fromarray(data)
        if self.transform:
            image = self.transform(image)
        return image, 0  # 返回图像和虚拟标签

def get_dataloader(config, train=True):
    """获取数据加载器"""
    if config.dataset == "mnist":
        dataset = MNISTDataset(
            root_dir=config.data_dir,
            train=train,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
    elif config.dataset == "emnist":
        dataset = EMNISTDataset(
            root_dir=config.data_dir,
            train=train,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
    elif config.dataset == "quickdraw":
        dataset = QuickDrawDataset(
            root_dir=config.data_dir,
            category="cat",  # 可以根据需要修改类别
            transform=transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=train,
        num_workers=config.data.num_workers,
        pin_memory=True
    ) 