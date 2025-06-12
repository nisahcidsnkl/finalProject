import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion.base import DiffusionModel
from utils.data_utils import get_dataloader

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_epoch(model, dataloader, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (data, _) in enumerate(progress_bar):
        data = data.to(device)
        
        # 生成随机时间步
        t = torch.randint(0, model.config.diffusion.num_timesteps, (data.shape[0],), device=device)
        
        # 添加噪声
        noise = torch.randn_like(data)
        noisy_data = model.add_noise(data, t, noise)
        
        # 预测噪声
        predicted_noise = model(noisy_data, t)
        
        # 计算损失
        loss = nn.MSELoss()(predicted_noise, noise)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 更新进度条
        progress_bar.set_postfix({'loss': loss.item()})
        
        # 记录到tensorboard
        if batch_idx % 100 == 0:
            writer.add_scalar('train/loss', loss.item(), epoch * len(dataloader) + batch_idx)
    
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = DiffusionModel(config).to(device)
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay
    )
    
    # 创建数据加载器
    train_loader = get_dataloader(config, train=True)
    
    # 创建tensorboard writer
    writer = SummaryWriter(config.logging.log_dir)
    
    # 创建检查点目录
    checkpoint_dir = os.path.join(config.logging.log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练循环
    for epoch in range(config.training.num_epochs):
        # 训练一个epoch
        avg_loss = train_epoch(model, train_loader, optimizer, device, epoch, writer)
        
        # 打印训练信息
        print(f'Epoch {epoch}: Average Loss = {avg_loss:.4f}')
        
        # 保存检查点
        if (epoch + 1) % config.training.save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
    
    writer.close()

if __name__ == '__main__':
    main() 