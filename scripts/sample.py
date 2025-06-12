import os
import sys
import yaml
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion.base import DiffusionModel

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_images(images, save_dir, epoch):
    """保存生成的图像"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 将图像转换为numpy数组
    images = images.cpu().numpy()
    
    # 创建图像网格
    n_images = min(16, len(images))
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(n_images):
        img = images[i].squeeze()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'samples_epoch_{epoch}.png'))
    plt.close()

def sample(model, device, n_samples=16):
    """从模型中采样生成图像"""
    model.eval()
    
    with torch.no_grad():
        # 从标准正态分布采样
        x = torch.randn(n_samples, model.config.model.in_channels, 
                       model.config.data.image_size, 
                       model.config.data.image_size).to(device)
        
        # 逐步去噪
        for t in tqdm(reversed(range(model.config.diffusion.num_timesteps)), desc='Sampling'):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            predicted_noise = model(x, t_batch)
            
            # 更新采样
            alpha = model.alphas[t]
            alpha_prev = model.alphas_prev[t]
            sigma = model.sigmas[t]
            
            noise = torch.randn_like(x) if t > 0 else 0
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha)) * predicted_noise) + sigma * noise
    
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--n_samples', type=int, default=16, help='Number of samples to generate')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = DiffusionModel(config).to(device)
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 生成图像
    samples = sample(model, device, args.n_samples)
    
    # 保存图像
    save_dir = os.path.join(config.logging.log_dir, 'images')
    save_images(samples, save_dir, checkpoint['epoch'])

if __name__ == '__main__':
    main() 