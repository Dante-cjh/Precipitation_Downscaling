import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from src.models import Network
from src.dataset.ddpm_dataset import RobustPrecipitationZScoreDataset

# --------------------------
# 计算二值指标函数（PC, PO, FAR）
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
    return PC, PO, FAR

# --------------------------
# 对单个样本进行评估
def evaluate_single_sample(pred_img, target_img, threshold=0.1):
    data_range = target_img.max() - target_img.min()  # 使用目标图像实际动态范围
    psnr_value = calculate_psnr(target_img, pred_img, data_range=data_range)
    ssim_value = compare_ssim(target_img, pred_img, data_range=data_range)
    rmse_value = np.sqrt(np.mean((pred_img - target_img) ** 2))
    PC, PO, FAR = calculate_metrics(pred_img, target_img, threshold)
    return psnr_value, ssim_value, rmse_value, PC, PO, FAR

# --------------------------
# 辅助函数：将数组归一化到 8 位灰度图像（对每幅图像独立做 min-max scaling）
def _normalize_to_8bit(array):
    arr_min, arr_max = array.min(), array.max()
    if np.isclose(arr_min, arr_max):
        return np.zeros_like(array, dtype=np.uint8)
    norm = (array - arr_min) / (arr_max - arr_min)
    norm_255 = (norm * 255).astype(np.uint8)
    return norm_255

# --------------------------
# UNet采样函数（无时间条件）
@torch.no_grad()
def sample_unet(input_batch, model, device, dataset):
    images_input = input_batch["inputs"].to(device)           # 7通道条件数据
    coarse = input_batch["coarse_acpcp"].to(device)             # 1通道低分辨率图
    fine = input_batch["fine_acpcp"].to(device)                 # 1通道真实高分辨率图
    # 模型输出预测残差
    residual = model(images_input, class_labels=None)
    # 通过数据集内置的逆归一化函数，将预测的 residual 与 coarse 相加恢复 fine 图像
    predicted = dataset.residual_to_fine_image(residual.detach().cpu(), coarse.cpu())
    return coarse.cpu(), fine.cpu(), predicted

# --------------------------
# 主评测流程
if __name__ == "__main__":
    # 设备选择
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # 模型选择：此处为 UNet
    modelname = "UNet"

    # 目录设置（请根据实际情况修改）
    data_dir = "../../processed_datasets/new_test/"      # 测试数据目录
    model_dir = "../../models/unet/"             # 模型保存目录
    save_dir = "../../predictions/unet_results/"
    os.makedirs(save_dir, exist_ok=True)

    # 加载 UNet 模型
    model = Network.UNet(
        img_resolution=(224, 224),
        in_channels=7,   # UNet模型输入为7通道条件数据
        out_channels=1,
        label_dim=0,
        use_diffuse=False
    ).to(device)
    model_path = os.path.join(model_dir, "best_model.pt")
    state_dict = torch.load(model_path, map_location=device)

    # 处理 state_dict，剥离多余的 "model." 前缀（如果存在）
    def fix_state_dict_keys(state_dict, extra_prefix="model."):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(extra_prefix):
                new_key = k[len(extra_prefix):]
            else:
                new_key = k
            new_state_dict[new_key] = v
        return new_state_dict

    fixed_state_dict = fix_state_dict_keys(state_dict)
    model.load_state_dict(fixed_state_dict)
    model.eval()

    # UNet采样函数
    sample_function = lambda batch, model, device, dataset, num_steps=None: sample_unet(batch, model, device, dataset)

    # 数据集加载
    norm_means = {'acpcp': 10.540719985961914, 'lsm': 0.31139194936462644, 'r2': 76.87171173095703, 't': 281.4097900390625, 'u10': 0.6429771780967712, 'v10': -0.13687361776828766, 'z': 2719.8411298253704}
    norm_stds = {'acpcp': 24.950708389282227, 'lsm': 0.44974926605736953, 'r2': 17.958908081054688, 't': 18.581451416015625, 'u10': 5.890835762023926, 'v10': 4.902953624725342, 'z': 6865.68850114493}
    residual_mean = 0.0020228675566613674
    residual_std = 6.931405067443848

    dataset_test = RobustPrecipitationZScoreDataset(data_dir, norm_means=norm_means, norm_stds=norm_stds, residual_mean=residual_mean, residual_std=residual_std)
    nlat, nlon = 224, 224
    ntime = len(dataset_test)
    print(f"Test samples: {ntime}, nlat: {nlat}, nlon: {nlon}")

    # 构建 DataLoader（保持顺序，shuffle=False）
    BATCH_SIZE = 8
    dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 预分配 numpy 数组保存预测结果（fine图）、coarse和真实fine图
    pred_array = np.zeros((ntime, nlat, nlon), dtype=np.float32)
    coarse_array = np.zeros((ntime, nlat, nlon), dtype=np.float32)
    fine_array = np.zeros((ntime, nlat, nlon), dtype=np.float32)

    t = 0
    for batch in tqdm(dataloader, desc="Inference"):
        coarse, fine, predicted = sample_function(batch, model, device, dataset_test, num_steps=None)
        bs = predicted.shape[0]
        pred_array[t:t+bs] = predicted.squeeze(1).detach().cpu().numpy()
        coarse_array[t:t+bs] = coarse.squeeze(1).detach().cpu().numpy()
        fine_array[t:t+bs] = fine.squeeze(1).detach().cpu().numpy()
        t += bs

    # ---------------------------
    # 保存灰度图：将 HR (真实 fine) 和 SR (预测 fine) 分别保存到对应文件夹
    total_psnr, total_ssim, total_rmse = 0, 0, 0
    total_PC, total_PO, total_FAR = 0, 0, 0
    num_samples = 0

    hr_dir = os.path.join(save_dir, "HR")
    sr_dir = os.path.join(save_dir, "SR")
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(sr_dir, exist_ok=True)

    for i in tqdm(range(ntime), desc="Evaluating samples"):
        pred_img = pred_array[i]
        target_img = fine_array[i]
        psnr_value, ssim_value, rmse_value, PC, PO, FAR = evaluate_single_sample(pred_img, target_img, threshold=0.1)
        print(f"Sample {i+1}: PSNR={psnr_value:.2f}, SSIM={ssim_value:.4f}, RMSE={rmse_value:.4f}, "
              f"PC={PC:.2f}%, PO={PO:.2f}%, FAR={FAR:.2f}%")
        total_psnr += psnr_value
        total_ssim += ssim_value
        total_rmse += rmse_value
        total_PC += PC
        total_PO += PO
        total_FAR += FAR
        num_samples += 1

        # 使用 _normalize_to_8bit 对每个图像做 min-max scaling 映射到 [0,255]
        hr_img = _normalize_to_8bit(fine_array[i])
        sr_img = _normalize_to_8bit(pred_array[i])
        hr_path = os.path.join(hr_dir, f"HR_{i+1:04d}.png")
        sr_path = os.path.join(sr_dir, f"SR_{i+1:04d}.png")
        cv2.imwrite(hr_path, hr_img)
        cv2.imwrite(sr_path, sr_img)

    print(f"\nAverage metrics over {num_samples} samples:")
    print(f"PSNR: {total_psnr/num_samples:.2f}, SSIM: {total_ssim/num_samples:.4f}, RMSE: {total_rmse/num_samples:.4f}")
    print(f"PC: {total_PC/num_samples:.2f}%, PO: {total_PO/num_samples:.2f}%, FAR: {total_FAR/num_samples:.2f}%")
    print(f"Saved HR images to {hr_dir} and SR images to {sr_dir}")