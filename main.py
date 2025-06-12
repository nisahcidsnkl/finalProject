# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import os
import argparse
import yaml
from scripts.train import main as train_main
from scripts.sample import main as sample_main
from scripts.evaluate import main as evaluate_main

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='扩散模型图像生成项目')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'sample', 'evaluate'],
                      help='运行模式：train（训练）, sample（采样）, evaluate（评估）')
    parser.add_argument('--config', type=str, required=True,
                      help='配置文件路径')
    parser.add_argument('--checkpoint', type=str,
                      help='模型检查点路径（用于sample和evaluate模式）')
    parser.add_argument('--n_samples', type=int, default=16,
                      help='生成样本数量（用于sample模式）')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建必要的目录
    os.makedirs(config.logging.log_dir, exist_ok=True)
    os.makedirs(os.path.join(config.logging.log_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config.logging.log_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(config.logging.log_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(config.logging.log_dir, 'plots'), exist_ok=True)
    
    if args.mode == 'train':
        train_main()
    elif args.mode == 'sample':
        if not args.checkpoint:
            raise ValueError("sample模式需要提供checkpoint参数")
        sample_main()
    elif args.mode == 'evaluate':
        if not args.checkpoint:
            raise ValueError("evaluate模式需要提供checkpoint参数")
        evaluate_main()

if __name__ == '__main__':
    main()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
