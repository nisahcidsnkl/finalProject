import os
import sys
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.models import inception_v3
from scipy import linalg
from PIL import Image
import torchvision.transforms as transforms

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cgan.cgan import CGAN
from utils.data_utils import get_dataloader

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class FID:
    def __init__(self, device):
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()
        
        # 移除最后的分类层
        self.model.fc = torch.nn.Identity()
        
        # 定义预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def get_features(self, images):
        """提取图像特征"""
        features = []
        with torch.no_grad():
            for img in images:
                # 转换为RGB
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                
                # 预处理
                img = self.preprocess(img).unsqueeze(0).to(self.device)
                
                # 提取特征
                feature = self.model(img)
                features.append(feature.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def calculate_statistics(self, features):
        """计算特征的均值和协方差"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def calculate_fid(self, mu1, sigma1, mu2, sigma2):
        """计算FID分数"""
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        # 检查数值稳定性
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

def calculate_mse_diversity(images):
    """计算生成图像的MSE多样性"""
    n_images = len(images)
    total_mse = 0
    count = 0
    
    for i in range(n_images):
        for j in range(i + 1, n_images):
            mse = torch.mean((images[i] - images[j]) ** 2).item()
            total_mse += mse
            count += 1
    
    return total_mse / count if count > 0 else 0

def evaluate(model, dataloader, fid, device, n_samples=1000):
    """评估模型性能"""
    model.generator.eval()
    
    # 生成样本
    print("Generating samples...")
    generated_images = []
    with torch.no_grad():
        for _ in tqdm(range(n_samples // 16)):
            labels = torch.randint(0, model.config.model.num_classes, (16,), device=device)
            samples = model.generate(labels)
            generated_images.extend(samples.cpu())
    
    # 获取真实图像
    print("Loading real images...")
    real_images = []
    for batch, _ in tqdm(dataloader):
        real_images.extend(batch)
        if len(real_images) >= n_samples:
            break
    
    # 计算FID
    print("Calculating FID...")
    real_features = fid.get_features(real_images[:n_samples])
    gen_features = fid.get_features(generated_images[:n_samples])
    
    real_mu, real_sigma = fid.calculate_statistics(real_features)
    gen_mu, gen_sigma = fid.calculate_statistics(gen_features)
    
    fid_score = fid.calculate_fid(real_mu, real_sigma, gen_mu, gen_sigma)
    
    # 计算MSE多样性
    print("Calculating MSE diversity...")
    mse_diversity = calculate_mse_diversity(generated_images[:n_samples])
    
    return {
        'fid': fid_score,
        'mse_diversity': mse_diversity
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = CGAN(config)
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.generator.load_state_dict(checkpoint['generator_state_dict'])
    model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # 创建数据加载器
    dataloader = get_dataloader(config, train=False)
    
    # 创建FID计算器
    fid = FID(device)
    
    # 评估模型
    metrics = evaluate(model, dataloader, fid, device)
    
    # 保存结果
    results_dir = os.path.join(config.logging.log_dir, 'metrics')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存FID分数
    fid_df = pd.DataFrame({
        'epoch': [checkpoint['epoch']],
        'fid_score': [metrics['fid']]
    })
    fid_df.to_csv(os.path.join(results_dir, 'fid_scores.csv'), mode='a', header=not os.path.exists(os.path.join(results_dir, 'fid_scores.csv')))
    
    # 保存MSE多样性
    mse_df = pd.DataFrame({
        'epoch': [checkpoint['epoch']],
        'mse_diversity': [metrics['mse_diversity']]
    })
    mse_df.to_csv(os.path.join(results_dir, 'mse_diversity.csv'), mode='a', header=not os.path.exists(os.path.join(results_dir, 'mse_diversity.csv')))
    
    print(f"Evaluation results:")
    print(f"FID Score: {metrics['fid']:.2f}")
    print(f"MSE Diversity: {metrics['mse_diversity']:.4f}")

if __name__ == '__main__':
    main() 