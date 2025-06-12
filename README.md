# 扩散模型图像生成项目

本项目实现了基于扩散模型的图像生成系统，包含多个模型变体，从基础扩散模型到完整的条件生成模型。

## 项目结构

```
finalProject/
├── data/                        # 数据集目录
├── models/                      # 模型定义
├── configs/                     # 配置文件
├── scripts/                     # 运行脚本
├── results/                     # 实验结果
├── utils/                       # 工具函数
└── main.py                      # 主入口
```

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (推荐)

## 安装

1. 克隆仓库：
```bash
git clone [repository_url]
cd finalProject
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
python scripts/train.py --config configs/mnist.yaml
```

### 生成图像

```bash
python scripts/sample.py --config configs/mnist.yaml --checkpoint path/to/checkpoint
```

### 评估模型

```bash
python scripts/evaluate.py --config configs/mnist.yaml --checkpoint path/to/checkpoint
```

## 模型变体

- A0: 基础扩散模型（无条件）
- A1: 条件扩散模型
- A2: 条件扩散+CLIP模型
- A3: 条件扩散+CLIP+CFG模型
- A4: 完整扩散模型（含注意力机制）

## 数据集

支持以下数据集：
- MNIST
- EMNIST
- QuickDraw

## 评估指标

- FID (Fréchet Inception Distance)
- MSE (Mean Squared Error)
- CLIP相似度
- 生成多样性

## 许可证

MIT License