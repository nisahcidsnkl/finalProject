import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from .base import TimeEmbedding, ResBlock

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.attn = Attention(dim, num_heads, dropout)
        
    def forward(self, x):
        return x + self.attn(self.norm(x))

class FullDiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 时间嵌入
        self.time_embedding = TimeEmbedding(config.model.time_embedding_size)
        
        # 标签嵌入
        self.label_embedding = nn.Embedding(config.model.num_classes, config.model.time_embedding_size)
        
        # 初始卷积层
        self.init_conv = nn.Conv2d(config.model.in_channels, config.model.hidden_size, 3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        current_channels = config.model.hidden_size
        for _ in range(3):
            self.down_blocks.append(nn.ModuleList([
                ResBlock(current_channels, current_channels * 2, config.model.time_embedding_size * 2),
                AttentionBlock(current_channels * 2)
            ]))
            current_channels *= 2
        
        # 中间块
        self.mid_block = nn.ModuleList([
            ResBlock(current_channels, current_channels, config.model.time_embedding_size * 2),
            AttentionBlock(current_channels),
            ResBlock(current_channels, current_channels, config.model.time_embedding_size * 2)
        ])
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        for _ in range(3):
            self.up_blocks.append(nn.ModuleList([
                ResBlock(current_channels, current_channels // 2, config.model.time_embedding_size * 2),
                AttentionBlock(current_channels // 2)
            ]))
            current_channels //= 2
        
        # 输出层
        self.final_norm = nn.GroupNorm(8, current_channels)
        self.final_conv = nn.Conv2d(current_channels, config.model.in_channels, 3, padding=1)

    def forward(self, x, t, labels):
        # 时间嵌入
        t_emb = self.time_embedding(t)
        
        # 标签嵌入
        label_emb = self.label_embedding(labels)
        
        # 拼接时间嵌入和标签嵌入
        emb = torch.cat([t_emb, label_emb], dim=-1)
        
        # 初始特征
        h = self.init_conv(x)
        
        # 下采样路径
        down_features = []
        for res_block, attn_block in self.down_blocks:
            h = res_block(h, emb)
            h = attn_block(h)
            down_features.append(h)
            h = F.avg_pool2d(h, 2)
        
        # 中间块
        h = self.mid_block[0](h, emb)
        h = self.mid_block[1](h)
        h = self.mid_block[2](h, emb)
        
        # 上采样路径
        for (res_block, attn_block), skip in zip(self.up_blocks, reversed(down_features)):
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            h = torch.cat([h, skip], dim=1)
            h = res_block(h, emb)
            h = attn_block(h)
        
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

    def remove_noise(self, x, t, labels):
        """从噪声图像中移除噪声"""
        predicted_noise = self(x, t, labels)
        
        # 计算噪声调度
        alpha = self.alphas[t]
        alpha_prev = self.alphas_prev[t]
        
        # 移除噪声
        x0 = (1 / torch.sqrt(alpha)[:, None, None, None]) * \
             (x - torch.sqrt(1 - alpha)[:, None, None, None] * predicted_noise)
        
        return x0 