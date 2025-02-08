import numpy as np
import torch

from src.eval import calculate_psnr

# 模拟预测和目标图像
pred = np.random.rand(224, 224)
target = np.random.rand(224, 224)

pred = torch.tensor(pred)
target = torch.tensor(target)

# 计算 PSNR
psnr = calculate_psnr(pred, target)
print(f"PSNR: {psnr}")