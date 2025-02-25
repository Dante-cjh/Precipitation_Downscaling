import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

# 读取两个 netCDF 文件
baseline_nc_file = "../../predictions/diffusion_unet/results/samples.nc"  # baseline 模型的文件
model_v2_nc_file = "../../predictions/diffusion_unet_v2_80/results/samples.nc"  # 新模型的文件

# 读取数据
baseline_ds = xr.open_dataset(baseline_nc_file)
model_v2_ds = xr.open_dataset(model_v2_nc_file)

# 获取 predicted 数据
baseline_predicted = baseline_ds["predicted"].values  # [time, lat, lon]
model_v2_predicted = model_v2_ds["predicted"].values  # [time, lat, lon]
fine = baseline_ds["fine"].values      # [time, lat, lon]
coarse = baseline_ds["coarse"].values  # [time, lat, lon]

# 确保 predicted 中的负值被置为 0
baseline_predicted[baseline_predicted < 0] = 0
model_v2_predicted[model_v2_predicted < 0] = 0

# 获取时间维度的大小
ntime = baseline_predicted.shape[0]  # 两个文件的 predicted 维度应相同

# 创建保存结果的文件夹
save_dir = "../../predictions/diffusion_unet_v2_80/results/compared_images/"
os.makedirs(save_dir, exist_ok=True)

# 遍历每个时间点，处理并保存图像
for t in tqdm(range(ntime)):
    # 获取当前时间的 baseline 和 new model predicted 图像
    baseline_predicted_img = baseline_predicted[t]
    model_v2_predicted_img = model_v2_predicted[t]
    fine_img = fine[t]
    coarse_img = coarse[t]

    # 获取实际数据的最小值和最大值
    data_min = min(coarse_img.min(), fine_img.min(), model_v2_predicted_img.min())
    data_max = max(coarse_img.max(), fine_img.max(), model_v2_predicted_img.max())

    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 显示 baseline 模型的 predicted 图像
    ax = axes[0, 0]
    c = ax.imshow(baseline_predicted_img, cmap='viridis', vmin=data_min, vmax=data_max)  # 归一化到 [0, 1] 范围
    ax.set_title(f"Baseline Predicted {t+1}")
    fig.colorbar(c, ax=ax)

    # 显示 model_v2 模型的 predicted 图像
    ax = axes[0, 1]
    c = ax.imshow(model_v2_predicted_img, cmap='viridis', vmin=data_min, vmax=data_max)
    ax.set_title(f"Model V2 Predicted {t+1}")
    fig.colorbar(c, ax=ax)

    # 显示 coarse 图
    ax = axes[1, 0]
    c = ax.imshow(coarse_img, cmap='viridis', vmin=data_min, vmax=data_max)  # 归一化到 [0, 1] 范围
    ax.set_title(f"Coarse {t + 1}")
    fig.colorbar(c, ax=ax)

    # 显示 fine 图
    ax = axes[1, 1]
    c = ax.imshow(fine_img, cmap='viridis', vmin=data_min, vmax=data_max)
    ax.set_title(f"Fine {t + 1}")
    fig.colorbar(c, ax=ax)


    # 保存合并后的图像
    combined_image_path = os.path.join(save_dir, f"combined_{t+1:04d}.png")
    plt.tight_layout()
    plt.savefig(combined_image_path)
    plt.close()

print("All images saved successfully.")