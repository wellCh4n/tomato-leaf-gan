import argparse
import os
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import sys

# 创建保存生成图像的目录
os.makedirs("generated_images", exist_ok=True)

# 设置参数
parser = argparse.ArgumentParser()
parser.add_argument("--n_images", type=int, default=10, help="生成图像的数量")
parser.add_argument("--model_path", type=str, default="models/best_generator.pth", help="生成器模型路径")
parser.add_argument("--latent_dim", type=int, default=1024, help="潜在空间的维度")
parser.add_argument("--img_size", type=int, default=256, help="图像尺寸")
parser.add_argument("--channels", type=int, default=3, help="图像通道数")
args = parser.parse_args()

# 检查CUDA是否可用
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# 定义生成器类 (与gan.py中相同)
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.init_size = args.img_size // 4  # 初始特征图大小
        self.l1 = nn.Sequential(nn.Linear(args.latent_dim, 128 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, args.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# 初始化生成器
generator = Generator()

# 加载模型权重
generator.load_state_dict(torch.load(args.model_path))

# 如果有CUDA，将模型移至GPU
if cuda:
    generator.cuda()

# 设置为评估模式
generator.eval()

print(f"正在使用模型 {args.model_path} 生成 {args.n_images} 张图像...")

# 生成图像
with torch.no_grad():
    for i in range(args.n_images):
        # 生成随机噪声
        z = Variable(Tensor(np.random.normal(0, 1, (1, args.latent_dim))))
        
        # 生成图像
        gen_img = generator(z)
        
        # 保存图像
        save_image(gen_img.data, f"generated_images/gen_img_{i+1}.png", normalize=True)
        print(f"已生成图像 {i+1}/{args.n_images}")

print("所有图像生成完成！保存在 'generated_images' 目录中")