import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os
import numpy as np
from PIL import Image

def get_transform(config, train=True):
    transform_list = [
        transforms.Resize(config['model']['image_size']),
        transforms.CenterCrop(config['model']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
    if train:
        transform_list.insert(2, transforms.RandomHorizontalFlip())
    return transforms.Compose(transform_list)

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

def get_mnist_dataset(config, train=True):
    transform = get_transform(config, train)
    dataset = datasets.MNIST(
        root=config['data']['train_path'] if train else config['data']['val_path'],
        train=train,
        download=True,
        transform=transform
    )
    return dataset

def get_custom_dataset(config, train=True):
    transform = get_transform(config, train)
    data_path = config['data']['train_path'] if train else config['data']['val_path']
    class CustomDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_files = []
            self.labels = []
            for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
                class_dir = os.path.join(root_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith(('.png', '.jpg', '.jpeg')):
                            self.image_files.append(os.path.join(class_dir, img_name))
                            self.labels.append(class_idx)
        def __len__(self):
            return len(self.image_files)
        def __getitem__(self, idx):
            img_path = self.image_files[idx]
            image = Image.open(img_path).convert('L')
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
    return CustomDataset(data_path, transform=transform)

def get_dataloader(config, train=True):
    """获取数据加载器"""
    if config['data']['dataset'].lower() == "mnist":
        dataset = get_mnist_dataset(config, train)
    else:
        dataset = get_custom_dataset(config, train)
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=train,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=train
    )
    return dataloader 