# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import os
import sys
import yaml
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter

from models.cgan.cgan import CGAN
from utils.data_utils import get_dataloader
from scripts.train_cgan import train_epoch
from scripts.evaluate_cgan import evaluate

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='CGAN Training and Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True, help='Training or evaluation mode')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint (required for evaluation)')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = CGAN(config)
    model = model.to(device)
    
    if args.mode == 'train':
        # 创建数据加载器
        train_loader = get_dataloader(config, train=True)
        
        # 创建TensorBoard写入器
        if config.logging.tensorboard:
            writer = SummaryWriter(config.logging.log_dir)
        else:
            writer = None
        
        # 创建检查点目录
        checkpoint_dir = os.path.join(config.logging.log_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 训练循环
        for epoch in range(config.training.num_epochs):
            print(f"\nEpoch {epoch+1}/{config.training.num_epochs}")
            
            # 训练一个epoch
            train_losses = train_epoch(model, train_loader, device, writer, epoch)
            
            # 打印训练损失
            print(f"Generator Loss: {train_losses['g_loss']:.4f}")
            print(f"Discriminator Loss: {train_losses['d_loss']:.4f}")
            
            # 保存检查点
            if (epoch + 1) % config.logging.save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'generator_state_dict': model.generator.state_dict(),
                    'discriminator_state_dict': model.discriminator.state_dict(),
                    'g_optimizer_state_dict': model.g_optimizer.state_dict(),
                    'd_optimizer_state_dict': model.d_optimizer.state_dict(),
                    'g_loss': train_losses['g_loss'],
                    'd_loss': train_losses['d_loss']
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        
        if writer:
            writer.close()
    
    elif args.mode == 'eval':
        if not args.checkpoint:
            raise ValueError("Checkpoint path is required for evaluation mode")
        
        # 加载检查点
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # 创建数据加载器
        eval_loader = get_dataloader(config, train=False)
        
        # 评估模型
        metrics = evaluate(model, eval_loader, device)
        
        # 打印评估结果
        print("\nEvaluation Results:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

if __name__ == '__main__':
    main()

