import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 标签嵌入
        self.label_embedding = nn.Embedding(config.model.num_classes, config.model.time_embedding_size)
        
        # 初始全连接层
        self.init_fc = nn.Linear(config.model.latent_dim + config.model.time_embedding_size, 
                               4 * 4 * config.model.hidden_size * 8)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList([
            # 4x4 -> 8x8
            nn.Sequential(
                nn.ConvTranspose2d(config.model.hidden_size * 8, config.model.hidden_size * 4, 4, 2, 1),
                nn.BatchNorm2d(config.model.hidden_size * 4),
                nn.ReLU()
            ),
            # 8x8 -> 16x16
            nn.Sequential(
                nn.ConvTranspose2d(config.model.hidden_size * 4, config.model.hidden_size * 2, 4, 2, 1),
                nn.BatchNorm2d(config.model.hidden_size * 2),
                nn.ReLU()
            ),
            # 16x16 -> 28x28
            nn.Sequential(
                nn.ConvTranspose2d(config.model.hidden_size * 2, config.model.hidden_size, 4, 2, 1),
                nn.BatchNorm2d(config.model.hidden_size),
                nn.ReLU()
            )
        ])
        
        # 输出层
        self.final_conv = nn.Conv2d(config.model.hidden_size, config.model.in_channels, 3, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, z, labels):
        # 标签嵌入
        label_emb = self.label_embedding(labels)
        
        # 拼接噪声和标签嵌入
        x = torch.cat([z, label_emb], dim=1)
        
        # 初始特征
        x = self.init_fc(x)
        x = x.view(-1, self.config.model.hidden_size * 8, 4, 4)
        
        # 上采样路径
        for block in self.up_blocks:
            x = block(x)
        
        # 输出层
        x = self.final_conv(x)
        x = self.tanh(x)
        
        return x

class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 标签嵌入
        self.label_embedding = nn.Embedding(config.model.num_classes, config.model.time_embedding_size)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList([
            # 28x28 -> 14x14
            nn.Sequential(
                nn.Conv2d(config.model.in_channels, config.model.hidden_size, 4, 2, 1),
                nn.LeakyReLU(0.2)
            ),
            # 14x14 -> 7x7
            nn.Sequential(
                nn.Conv2d(config.model.hidden_size, config.model.hidden_size * 2, 4, 2, 1),
                nn.BatchNorm2d(config.model.hidden_size * 2),
                nn.LeakyReLU(0.2)
            ),
            # 7x7 -> 4x4
            nn.Sequential(
                nn.Conv2d(config.model.hidden_size * 2, config.model.hidden_size * 4, 4, 2, 1),
                nn.BatchNorm2d(config.model.hidden_size * 4),
                nn.LeakyReLU(0.2)
            )
        ])
        
        # 标签条件层
        self.label_condition = nn.Sequential(
            nn.Linear(config.model.time_embedding_size, 4 * 4),
            nn.LeakyReLU(0.2)
        )
        
        # 输出层
        self.final_conv = nn.Conv2d(config.model.hidden_size * 4 + 1, 1, 4, 1, 0)
        
    def forward(self, x, labels):
        # 标签嵌入
        label_emb = self.label_embedding(labels)
        
        # 下采样路径
        for block in self.down_blocks:
            x = block(x)
        
        # 标签条件
        label_condition = self.label_condition(label_emb)
        label_condition = label_condition.view(-1, 1, 4, 4)
        
        # 拼接特征和标签条件
        x = torch.cat([x, label_condition], dim=1)
        
        # 输出层
        x = self.final_conv(x)
        x = x.view(-1, 1)
        
        return x

class CGAN:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建生成器和判别器
        self.generator = Generator(config).to(self.device)
        self.discriminator = Discriminator(config).to(self.device)
        
        # 创建优化器
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.training.learning_rate,
            betas=(config.training.beta1, config.training.beta2)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.training.learning_rate,
            betas=(config.training.beta1, config.training.beta2)
        )
        
        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss()
        
    def train_step(self, real_images, labels):
        batch_size = real_images.size(0)
        
        # 真实标签和虚假标签
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # 训练判别器
        self.d_optimizer.zero_grad()
        
        # 真实图像的判别器输出
        real_outputs = self.discriminator(real_images, labels)
        d_loss_real = self.criterion(real_outputs, real_labels)
        
        # 生成虚假图像
        z = torch.randn(batch_size, self.config.model.latent_dim).to(self.device)
        fake_images = self.generator(z, labels)
        
        # 虚假图像的判别器输出
        fake_outputs = self.discriminator(fake_images.detach(), labels)
        d_loss_fake = self.criterion(fake_outputs, fake_labels)
        
        # 判别器总损失
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        # 训练生成器
        self.g_optimizer.zero_grad()
        
        # 重新生成虚假图像
        fake_images = self.generator(z, labels)
        fake_outputs = self.discriminator(fake_images, labels)
        
        # 生成器损失
        g_loss = self.criterion(fake_outputs, real_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item()
        }
    
    def generate(self, labels):
        """生成图像"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(labels.size(0), self.config.model.latent_dim).to(self.device)
            fake_images = self.generator(z, labels)
        return fake_images 