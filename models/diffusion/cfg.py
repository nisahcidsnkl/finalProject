import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from .base import TimeEmbedding, ResBlock
from .conditional import ConditionalDiffusionModel

class CFGDiffusionModel(ConditionalDiffusionModel):
    def __init__(self, config):
        super().__init__(config)
        self.cfg_scale = config.model.cfg_scale
        
    def forward(self, x, t, labels, cfg_scale=None):
        """使用Classifier-Free Guidance的前向传播"""
        if cfg_scale is None:
            cfg_scale = self.cfg_scale
            
        # 无条件预测
        uncond_labels = torch.zeros_like(labels)
        uncond_pred = super().forward(x, t, uncond_labels)
        
        # 条件预测
        cond_pred = super().forward(x, t, labels)
        
        # 应用CFG
        return uncond_pred + cfg_scale * (cond_pred - uncond_pred)
    
    def remove_noise(self, x, t, labels, cfg_scale=None):
        """使用CFG的噪声移除"""
        if cfg_scale is None:
            cfg_scale = self.cfg_scale
            
        # 无条件预测
        uncond_labels = torch.zeros_like(labels)
        uncond_pred = super().remove_noise(x, t, uncond_labels)
        
        # 条件预测
        cond_pred = super().remove_noise(x, t, labels)
        
        # 应用CFG
        return uncond_pred + cfg_scale * (cond_pred - uncond_pred)
    
    def sample(self, batch_size, device, labels, cfg_scale=None):
        """使用CFG进行采样"""
        if cfg_scale is None:
            cfg_scale = self.cfg_scale
            
        # 从标准正态分布采样
        x = torch.randn(batch_size, self.config.model.in_channels, 
                       self.config.data.image_size, 
                       self.config.data.image_size).to(device)
        
        # 逐步去噪
        for t in reversed(range(self.config.diffusion.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 使用CFG进行预测
            predicted_noise = self(x, t_batch, labels, cfg_scale)
            
            # 更新采样
            alpha = self.alphas[t]
            alpha_prev = self.alphas_prev[t]
            sigma = self.sigmas[t]
            
            noise = torch.randn_like(x) if t > 0 else 0
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha)) * predicted_noise) + sigma * noise
        
        return x 