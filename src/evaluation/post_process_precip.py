import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr

# 读取 netCDF 文件
nc_file = "../../predictions/diffusion_unet_v2/results/samples.nc"
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
save_dir = "../../predictions/diffusion_unet_v2/results/combined_images_origin/"
os.makedirs(save_dir, exist_ok=True)

# 初始化用于计算平均值的指标
total_psnr, total_ssim, total_rmse = 0, 0, 0
total_PC, total_PO, total_FAR = 0, 0, 0
total_POD, total_HSS = 0, 0
num_samples = 0

# --------------------------
# 计算二值指标函数（PC, PO, FAR, PD, HSS）
def calculate_metrics(pred, target, threshold=0.1):
    pred_binary = (pred > threshold).astype(int)
    target_binary = (target > threshold).astype(int)
    NA = ((pred_binary == 1) & (target_binary == 1)).sum()
    NB = ((pred_binary == 1) & (target_binary == 0)).sum()
    NC = ((pred_binary == 0) & (target_binary == 1)).sum()
    ND = ((pred_binary == 0) & (target_binary == 0)).sum()
    PC = (NA + ND) / (NA + NB + NC + ND) * 100 if (NA + NB + NC + ND) > 0 else 0
    PO = NC / (NA + NC) * 100 if (NA + NC) > 0 else 0
    FAR = NB / (NA + NB) * 100 if (NA + NB) > 0 else 0
    POD = NA / (NA + NC) * 100 if (NA + NC) > 0 else 0
    numerator = 2 * (NA * ND - NB * NC)
    denominator = (NA + NC) * (NB + ND) + (NA + NB) * (NC + ND)
    HSS = numerator / denominator if denominator != 0 else 0.0
    return PC, PO, FAR, POD, HSS

# --------------------------
# 对单个样本进行评估
def evaluate_single_sample(pred_img, target_img, threshold=0.1):
    data_range = target_img.max() - target_img.min()  # 使用目标图像实际动态范围
    psnr_value = calculate_psnr(target_img, pred_img, data_range=data_range)
    ssim_value = compare_ssim(target_img, pred_img, data_range=data_range)
    rmse_value = np.sqrt(np.mean((pred_img - target_img) ** 2))
    PC, PO, FAR, POD, HSS = calculate_metrics(pred_img, target_img, threshold)
    return psnr_value, ssim_value, rmse_value, PC, PO, FAR, POD, HSS

# 遍历每个时间点，处理并保存图像
for t in tqdm(range(ntime)):
    # 获取当前时间的coarse, fine, predicted图像
    coarse_img = coarse[t]
    fine_img = fine[t]
    predicted_img = predicted[t]

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

    # 评估当前样本
    psnr_value, ssim_value, rmse_value, PC, PO, FAR, POD, HSS = evaluate_single_sample(predicted_img, fine_img,
                                                                                       threshold=0.1)

    # 累加指标值
    total_psnr += psnr_value
    total_ssim += ssim_value
    total_rmse += rmse_value
    total_PC += PC
    total_PO += PO
    total_FAR += FAR
    total_POD += POD
    total_HSS += HSS
    num_samples += 1

# 输出平均指标值
print(f"\nAverage metrics over {num_samples} samples:")
print(f"PSNR: {total_psnr/num_samples:.2f}")
print(f"SSIM: {total_ssim/num_samples:.4f}")
print(f"RMSE: {total_rmse/num_samples:.4f}")
print(f"PC: {total_PC/num_samples:.2f}%")
print(f"PO: {total_PO/num_samples:.2f}%")
print(f"FAR: {total_FAR/num_samples:.2f}%")
print(f"POD: {total_POD/num_samples:.2f}%")
print(f"HSS: {total_HSS/num_samples:.4f}")

print("All images saved successfully.")