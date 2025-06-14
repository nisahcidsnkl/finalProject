import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from .base import TimeEmbedding, ResBlock
import clip

class CLIPConditionalDiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 加载CLIP模型
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
        self.clip_model.eval()
        
        # 冻结CLIP参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # 时间嵌入
        self.time_embedding = TimeEmbedding(config.model.time_embedding_size)
        
        # CLIP特征投影
        self.clip_projection = nn.Sequential(
            nn.Linear(512, config.model.time_embedding_size),
            nn.LayerNorm(config.model.time_embedding_size),
            nn.SiLU(),
            nn.Linear(config.model.time_embedding_size, config.model.time_embedding_size)
        )
        
        # 初始卷积层
        self.init_conv = nn.Conv2d(config.model.in_channels, config.model.hidden_size, 3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        current_channels = config.model.hidden_size
        for _ in range(3):
            self.down_blocks.append(
                ResBlock(current_channels, current_channels * 2, config.model.time_embedding_size * 2)
            )
            current_channels *= 2
        
        # 中间块
        self.mid_block = ResBlock(current_channels, current_channels, config.model.time_embedding_size * 2)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        for _ in range(3):
            self.up_blocks.append(
                ResBlock(current_channels, current_channels // 2, config.model.time_embedding_size * 2)
            )
            current_channels //= 2
        
        # 输出层
        self.final_norm = nn.GroupNorm(8, current_channels)
        self.final_conv = nn.Conv2d(current_channels, config.model.in_channels, 3, padding=1)

    def encode_text(self, text):
        """使用CLIP编码文本"""
        with torch.no_grad():
            text_tokens = clip.tokenize(text).to(next(self.parameters()).device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, x, t, text_features):
        # 时间嵌入
        t_emb = self.time_embedding(t)
        
        # CLIP特征投影
        clip_emb = self.clip_projection(text_features)
        
        # 拼接时间嵌入和CLIP嵌入
        emb = torch.cat([t_emb, clip_emb], dim=-1)
        
        # 初始特征
        h = self.init_conv(x)
        
        # 下采样路径
        down_features = []
        for block in self.down_blocks:
            h = block(h, emb)
            down_features.append(h)
            h = F.avg_pool2d(h, 2)
        
        # 中间块
        h = self.mid_block(h, emb)
        
        # 上采样路径
        for block in self.up_blocks:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            h = torch.cat([h, down_features.pop()], dim=1)
            h = block(h, emb)
        
        # 输出层
        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_conv(h)
        
        return h

    def add_noise(self, x, t, noise=None):
        """添加噪声到输入图像"""
        if noise is None:
            noise = torch.randn_like(x)
        
        # 计算噪声调度
        alpha = self.alphas[t]
        alpha_prev = self.alphas_prev[t]
        
        # 添加噪声
        noisy_x = torch.sqrt(alpha)[:, None, None, None] * x + \
                 torch.sqrt(1 - alpha)[:, None, None, None] * noise
        
        return noisy_x

    def remove_noise(self, x, t, text_features):
        """从噪声图像中移除噪声"""
        predicted_noise = self(x, t, text_features)
        
        # 计算噪声调度
        alpha = self.alphas[t]
        alpha_prev = self.alphas_prev[t]
        
        # 移除噪声
        x0 = (1 / torch.sqrt(alpha)[:, None, None, None]) * \
             (x - torch.sqrt(1 - alpha)[:, None, None, None] * predicted_noise)
        
        return x0 