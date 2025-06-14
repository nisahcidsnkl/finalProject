import os
import sys
import yaml
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion.conditional import ConditionalDiffusionModel
from models.diffusion.clip_conditional import CLIPConditionalDiffusionModel
from models.diffusion.cfg import CFGDiffusionModel
from models.diffusion.attention import FullDiffusionModel
from utils.data_utils import get_dataloader
from utils.clip_utils import get_clip_processor

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_epoch(model, dataloader, device, writer, epoch, clip_processor=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        
        # 根据模型类型处理输入
        if isinstance(model, CLIPConditionalDiffusionModel):
            # 对于CLIP模型，需要将标签转换为文本描述
            text_descriptions = [f"class {label.item()}" for label in labels]
            text_features = clip_processor.get_text_embeddings(text_descriptions)
            loss = model.train_step(images, text_features)
        else:
            # 对于其他模型，直接使用标签
            loss = model.train_step(images, labels)
        
        total_loss += loss.item()
        
        # 更新进度条
        progress_bar.set_postfix({'loss': loss.item()})
        
        # 记录到TensorBoard
        if writer and batch_idx % 100 == 0:
            writer.add_scalar('train/loss', loss.item(), epoch * len(dataloader) + batch_idx)
    
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_type', type=str, required=True, 
                      choices=['conditional', 'clip', 'cfg', 'attention'],
                      help='Type of diffusion model to train')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    if args.model_type == 'conditional':
        model = ConditionalDiffusionModel(config)
    elif args.model_type == 'clip':
        model = CLIPConditionalDiffusionModel(config)
        clip_processor = get_clip_processor()
    elif args.model_type == 'cfg':
        model = CFGDiffusionModel(config)
    elif args.model_type == 'attention':
        model = FullDiffusionModel(config)
    
    model = model.to(device)
    
    # 创建数据加载器
    train_loader = get_dataloader(config, train=True)
    
    # 创建TensorBoard写入器
    if config.logging.tensorboard:
        writer = SummaryWriter(os.path.join(config.logging.log_dir, args.model_type))
    else:
        writer = None
    
    # 创建检查点目录
    checkpoint_dir = os.path.join(config.logging.log_dir, args.model_type, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练循环
    for epoch in range(config.training.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.training.num_epochs}")
        
        # 训练一个epoch
        if args.model_type == 'clip':
            avg_loss = train_epoch(model, train_loader, device, writer, epoch, clip_processor)
        else:
            avg_loss = train_epoch(model, train_loader, device, writer, epoch)
        
        print(f"Average Loss: {avg_loss:.4f}")
        
        # 保存检查点
        if (epoch + 1) % config.logging.save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    if writer:
        writer.close()

if __name__ == '__main__':
    main() 