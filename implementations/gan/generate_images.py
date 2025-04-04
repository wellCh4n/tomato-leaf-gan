import argparse
import os
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from gan import Generator, opt

# 创建保存生成图像的目录
os.makedirs("generated_images", exist_ok=True)

# 设置参数
parser = argparse.ArgumentParser()
parser.add_argument("--n_images", type=int, default=10, help="生成图像的数量")
parser.add_argument("--model_path", type=str, default="models/best_generator.pth", help="生成器模型路径")
args = parser.parse_args()

# 检查CUDA是否可用
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

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
        z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
        
        # 生成图像
        gen_img = generator(z)
        
        # 保存图像
        save_image(gen_img.data, f"generated_images/gen_img_{i+1}.png", normalize=True)
        print(f"已生成图像 {i+1}/{args.n_images}")

print("所有图像生成完成！保存在 'generated_images' 目录中")