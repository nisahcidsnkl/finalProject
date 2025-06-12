import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_curves(log_dir, save_dir):
    """绘制训练损失曲线"""
    # 读取tensorboard日志
    events_file = os.path.join(log_dir, 'events.out.tfevents.*')
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置绘图风格
    plt.style.use('seaborn')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))
    plt.close()

def plot_metrics(metrics_dir, save_dir):
    """绘制评估指标曲线"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 读取指标数据
    fid_scores = pd.read_csv(os.path.join(metrics_dir, 'fid_scores.csv'))
    mse_diversity = pd.read_csv(os.path.join(metrics_dir, 'mse_diversity.csv'))
    
    # 设置绘图风格
    plt.style.use('seaborn')
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制FID分数
    ax1.plot(fid_scores['epoch'], fid_scores['fid_score'], 'b-', label='FID Score')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('FID Score')
    ax1.set_title('FID Score over Training')
    ax1.grid(True)
    ax1.legend()
    
    # 绘制MSE多样性
    ax2.plot(mse_diversity['epoch'], mse_diversity['mse_diversity'], 'r-', label='MSE Diversity')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Diversity')
    ax2.set_title('MSE Diversity over Training')
    ax2.grid(True)
    ax2.legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'metrics.png'))
    plt.close()

def plot_samples(images_dir, save_dir, n_samples=16):
    """绘制生成的样本图像"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取所有样本图像
    sample_files = sorted([f for f in os.listdir(images_dir) if f.startswith('samples_epoch_')])
    
    # 选择最新的n_samples个样本
    sample_files = sample_files[-n_samples:]
    
    # 创建图像网格
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, sample_file in enumerate(sample_files):
        if i >= 16:
            break
            
        # 读取图像
        img = plt.imread(os.path.join(images_dir, sample_file))
        
        # 显示图像
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        
        # 添加epoch信息
        epoch = int(sample_file.split('_')[-1].split('.')[0])
        axes[i].set_title(f'Epoch {epoch}')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'sample_progression.png'))
    plt.close()

def plot_comparison(real_images, generated_images, save_dir):
    """绘制真实图像和生成图像的对比"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建图像网格
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    
    # 显示真实图像
    for i in range(4):
        for j in range(4):
            axes[i, j].imshow(real_images[i*4 + j].squeeze(), cmap='gray')
            axes[i, j].axis('off')
            if i == 0 and j == 0:
                axes[i, j].set_title('Real Images')
    
    # 显示生成图像
    for i in range(4):
        for j in range(4):
            axes[i, j+4].imshow(generated_images[i*4 + j].squeeze(), cmap='gray')
            axes[i, j+4].axis('off')
            if i == 0 and j == 0:
                axes[i, j+4].set_title('Generated Images')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'comparison.png'))
    plt.close() 