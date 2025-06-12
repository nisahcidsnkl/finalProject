import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class TimeEmbedding(nn.Module):
    def __init__(self, time_embedding_size):
        super().__init__()
        self.time_embedding_size = time_embedding_size
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_size, time_embedding_size * 4),
            nn.SiLU(),
            nn.Linear(time_embedding_size * 4, time_embedding_size)
        )

    def forward(self, t):
        # 将时间步转换为正弦位置编码
        half_dim = self.time_embedding_size // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return self.time_mlp(embeddings)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_size, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_size, out_channels)
        )
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        # 添加时间嵌入
        h = h + self.time_mlp(t)[:, :, None, None]
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)

class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 时间嵌入
        self.time_embedding = TimeEmbedding(config.model.time_embedding_size)
        
        # 初始卷积层
        self.init_conv = nn.Conv2d(config.model.in_channels, config.model.hidden_size, 3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        current_channels = config.model.hidden_size
        for _ in range(3):
            self.down_blocks.append(
                ResBlock(current_channels, current_channels * 2, config.model.time_embedding_size)
            )
            current_channels *= 2
        
        # 中间块
        self.mid_block = ResBlock(current_channels, current_channels, config.model.time_embedding_size)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        for _ in range(3):
            self.up_blocks.append(
                ResBlock(current_channels, current_channels // 2, config.model.time_embedding_size)
            )
            current_channels //= 2
        
        # 输出层
        self.final_norm = nn.GroupNorm(8, current_channels)
        self.final_conv = nn.Conv2d(current_channels, config.model.in_channels, 3, padding=1)

    def forward(self, x, t):
        # 时间嵌入
        t = self.time_embedding(t)
        
        # 初始特征
        h = self.init_conv(x)
        
        # 下采样路径
        down_features = []
        for block in self.down_blocks:
            h = block(h, t)
            down_features.append(h)
            h = F.avg_pool2d(h, 2)
        
        # 中间块
        h = self.mid_block(h, t)
        
        # 上采样路径
        for block in self.up_blocks:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            h = torch.cat([h, down_features.pop()], dim=1)
            h = block(h, t)
        
        # 输出层
        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_conv(h)
        
        return h