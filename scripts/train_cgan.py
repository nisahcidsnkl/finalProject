import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cgan.cgan import CGAN
from utils.data_utils import get_dataloader

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_epoch(model, dataloader, epoch, writer):
    model.generator.train()
    model.discriminator.train()
    total_d_loss = 0
    total_g_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (data, labels) in enumerate(progress_bar):
        data = data.to(model.device)
        labels = labels.to(model.device)
        
        # 训练一个批次
        losses = model.train_step(data, labels)
        
        total_d_loss += losses['d_loss']
        total_g_loss += losses['g_loss']
        
        # 更新进度条
        progress_bar.set_postfix({
            'd_loss': losses['d_loss'],
            'g_loss': losses['g_loss']
        })
        
        # 记录到tensorboard
        if batch_idx % 100 == 0:
            writer.add_scalar('train/d_loss', losses['d_loss'], epoch * len(dataloader) + batch_idx)
            writer.add_scalar('train/g_loss', losses['g_loss'], epoch * len(dataloader) + batch_idx)
    
    return total_d_loss / len(dataloader), total_g_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建模型
    model = CGAN(config)
    
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
        avg_d_loss, avg_g_loss = train_epoch(model, train_loader, epoch, writer)
        
        # 打印训练信息
        print(f'Epoch {epoch}:')
        print(f'Average D Loss = {avg_d_loss:.4f}')
        print(f'Average G Loss = {avg_g_loss:.4f}')
        
        # 保存检查点
        if (epoch + 1) % config.training.save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': model.generator.state_dict(),
                'discriminator_state_dict': model.discriminator.state_dict(),
                'g_optimizer_state_dict': model.g_optimizer.state_dict(),
                'd_optimizer_state_dict': model.d_optimizer.state_dict(),
                'd_loss': avg_d_loss,
                'g_loss': avg_g_loss,
            }, checkpoint_path)
    
    writer.close()

if __name__ == '__main__':
    main() 