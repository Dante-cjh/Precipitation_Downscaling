import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

# 读取 netCDF 文件
nc_file = "../../predictions/diffusion_unet/results/samples.nc"
ds = xr.open_dataset(nc_file)

# 获取数据
coarse = ds["coarse"].values  # [time, lat, lon]
fine = ds["fine"].values      # [time, lat, lon]
predicted = ds["predicted"].values  # [time, lat, lon]

# 确保 predicted 中的负值被置为 0
predicted[predicted < 0] = 0

# 获取时间维度的大小
ntime = coarse.shape[0]  # 或 fine.shape[0] 或 predicted.shape[0]，它们应该是一样的

# 生成保存结果的文件夹
save_dir = "../../predictions/diffusion_unet/results/combined_images_origin/"
os.makedirs(save_dir, exist_ok=True)

# 遍历每个时间点，处理并保存图像
for t in tqdm(range(ntime)):
    # 获取当前时间的coarse, fine, predicted图像
    coarse_img = coarse[t]
    fine_img = fine[t]
    predicted_img = predicted[t]

    # # 归一化到 [0, 1] 范围
    # coarse_img_norm = (coarse_img - coarse_img.min()) / (coarse_img.max() - coarse_img.min())
    # fine_img_norm = (fine_img - fine_img.min()) / (fine_img.max() - fine_img.min())
    # predicted_img_norm = (predicted_img - predicted_img.min()) / (predicted_img.max() - predicted_img.min())

    # 获取实际数据的最小值和最大值
    data_min = min(coarse_img.min(), fine_img.min(), predicted_img.min())
    data_max = max(coarse_img.max(), fine_img.max(), predicted_img.max())

    # 创建一个3x1的子图（垂直排列）
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 显示 coarse 图
    ax = axes[0]
    c = ax.imshow(coarse_img, cmap='viridis', vmin=data_min, vmax=data_max)  # 归一化到 [0, 1] 范围
    ax.set_title(f"Coarse {t+1}")
    fig.colorbar(c, ax=ax)

    # 显示 fine 图
    ax = axes[1]
    c = ax.imshow(fine_img, cmap='viridis', vmin=data_min, vmax=data_max)
    ax.set_title(f"Fine {t+1}")
    fig.colorbar(c, ax=ax)

    # 显示 predicted 图
    ax = axes[2]
    c = ax.imshow(predicted_img, cmap='viridis', vmin=data_min, vmax=data_max)
    ax.set_title(f"Predicted {t+1}")
    fig.colorbar(c, ax=ax)

    # 保存合并后的图像
    combined_image_path = os.path.join(save_dir, f"combined_{t+1:04d}.png")
    plt.tight_layout()
    plt.savefig(combined_image_path)
    plt.close()

print("All images saved successfully.")