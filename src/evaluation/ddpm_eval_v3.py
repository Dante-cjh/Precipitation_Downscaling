import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
import xarray as xr

from src.models import NetworkV3
from src.dataset.ddpm_dataset import RobustPrecipitationZScoreDataset


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
    images_input = input_batch["inputs"].to(device)  # 7通道条件数据
    coarse = input_batch["coarse_acpcp"].to(device)  # 1通道低分辨率图
    fine = input_batch["fine_acpcp"].to(device)  # 1通道真实高分辨率图
    # 模型输出预测残差
    residual = model(images_input, class_labels=None)
    # 通过数据集内置的逆归一化函数，将预测的 residual 与 coarse 相加恢复 fine 图像
    predicted = dataset.residual_to_fine_image(residual.detach().cpu(), coarse.cpu())
    return coarse.cpu(), fine.cpu(), predicted


# --------------------------
# 扩散模型采样函数
@torch.no_grad()
def sample_diffusion(input_batch, model, device, dataset, num_steps=40,
                     sigma_min=0.002, sigma_max=80, rho=7, S_churn=40,
                     S_min=0, S_max=float('inf'), S_noise=1):
    # 反归一化
    inverse_normalize_residual = lambda residual_norm: ((residual_norm * residual_std) + residual_mean)

    images_input = input_batch["inputs"].to(device)  # 7通道条件数据
    coarse = input_batch["coarse_acpcp"].to(device)  # 1通道低分辨率图
    fine = input_batch["fine_acpcp"].to(device)  # 1通道真实高分辨率图

    if hasattr(model, "sigma_min"):
        sigma_min = max(sigma_min, model.sigma_min)
    if hasattr(model, "sigma_max"):
        sigma_max = min(sigma_max, model.sigma_max)

    init_noise = torch.randn((images_input.shape[0], 1, images_input.shape[2],
                              images_input.shape[3]),
                             dtype=torch.float64, device=device)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=init_noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    if hasattr(model, "round_sigma"):
        t_steps = torch.cat([model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    else:
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    x_next = init_noise * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if (S_min <= t_cur <= S_max) else 0
        if hasattr(model, "round_sigma"):
            t_hat = model.round_sigma(t_cur + gamma * t_cur)
        else:
            t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + ((t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur))
        denoised = model(x_hat, t_hat, images_input).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            denoised = model(x_next, t_next, images_input).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    predicted_precipitation = coarse.cpu() + inverse_normalize_residual(x_next.detach().cpu())
    return coarse.cpu(), fine.cpu(), predicted_precipitation


# --------------------------
# 主评测流程
if __name__ == "__main__":
    # 设备选择
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # 模型选择：可选 "Diffusion" 或 "UNet" 或 "LinearInterpolation"
    modelname = "Diffusion"

    # 目录设置（请根据实际情况修改）
    data_dir = "../../processed_datasets/new_test/"  # 测试数据目录
    model_dir = "../../models/unet_diffusion_v3/"  # 模型保存目录
    save_dir = "../../predictions/diffusion_unet_v3/results/"
    os.makedirs(save_dir, exist_ok=True)

    # 根据模型类型加载模型和采样函数
    if modelname == "Diffusion":
        model = NetworkV3.EDMPrecond(
            img_resolution=(224, 224),
            in_channels=8,  # 7条件通道 + 1残差通道
            out_channels=1,
            label_dim=0,
            use_diffuse=True
        ).to(device)
        model_path = os.path.join(model_dir, "best_model.pt")
        state_dict = torch.load(model_path, map_location=device, weights_only=True)

        # 处理 state_dict，剥离多余的 "model." 前缀（如果存在）
        def fix_state_dict_keys(state_dict, extra_prefix="model."):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith(extra_prefix + "model."):
                    new_key = k[len(extra_prefix):]
                else:
                    new_key = k
                new_state_dict[new_key] = v
            return new_state_dict


        fixed_state_dict = fix_state_dict_keys(state_dict)
        model.load_state_dict(state_dict)
        model.eval()  # 确保模型进入评估模式
        sample_function = lambda batch, model, device, dataset, num_steps=100: sample_diffusion(batch, model, device,
                                                                                                dataset,
                                                                                                num_steps=num_steps)
        num_steps = 100
        rngs = range(0, 30)
    elif modelname == "UNet":
        model = NetworkV3.UNet(
            img_resolution=(224, 224),
            in_channels=7,  # UNet模型输入为7通道条件数据
            out_channels=1,
            label_dim=0,
            use_diffuse=False
        ).to(device)
        model_path = os.path.join(model_dir, "unet.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        sample_function = lambda batch, model, device, dataset, num_steps=None: sample_unet(batch, model, device,
                                                                                            dataset)
        num_steps = None
        rngs = [""]
    elif modelname == "LinearInterpolation":
        def sample_function(batch, model, device, dataset, num_steps=None):
            coarse = batch["coarse_acpcp"]
            fine = batch["fine_acpcp"]
            return coarse, fine, coarse
        model = None
        num_steps = None
        rngs = [""]
    else:
        raise Exception(f"Choose modelname either Diffusion or UNet. You chose {modelname}")

    print(f"Running model {modelname} with sample function {sample_function}.")

    # 如果你预先计算了归一化参数，可以将它们传入数据集构造函数
    norm_means = {'acpcp': 10.540719985961914, 'lsm': 0.31139194936462644, 'r2': 76.87171173095703,
                  't': 281.4097900390625, 'u10': 0.6429771780967712, 'v10': -0.13687361776828766,
                  'z': 2719.8411298253704}
    norm_stds = {'acpcp': 24.950708389282227, 'lsm': 0.44974926605736953, 'r2': 17.958908081054688,
                 't': 18.581451416015625, 'u10': 5.890835762023926, 'v10': 4.902953624725342, 'z': 6865.68850114493}
    residual_mean = 0.0020228675566613674
    residual_std = 6.931405067443848

    dataset_test = RobustPrecipitationZScoreDataset(data_dir, norm_means=norm_means, norm_stds=norm_stds,
                                                    residual_mean=residual_mean, residual_std=residual_std)
    if hasattr(dataset_test, "lat") and hasattr(dataset_test, "lon"):
        nlat = dataset_test.lat.shape[0]
        nlon = dataset_test.lon.shape[0]
    else:
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
        if model is not None:
            coarse, fine, predicted = sample_function(batch, model, device, dataset_test, num_steps=num_steps)
        else:
            coarse, fine, predicted = sample_function(batch, None, device, dataset_test, num_steps=num_steps)
        bs = predicted.shape[0]
        pred_array[t:t + bs] = predicted.squeeze(1).detach().cpu().numpy()
        coarse_array[t:t + bs] = coarse.squeeze(1).detach().cpu().numpy()
        fine_array[t:t + bs] = fine.squeeze(1).detach().cpu().numpy()
        t += bs

    # ---------------------------
    # 保存预测结果为 netCDF 文件
    ds = xr.Dataset({
        "predicted": (("time", "lat", "lon"), pred_array),
        "coarse": (("time", "lat", "lon"), coarse_array),
        "fine": (("time", "lat", "lon"), fine_array),
    })
    if hasattr(dataset_test, "lat") and hasattr(dataset_test, "lon"):
        ds = ds.assign_coords(lat=dataset_test.lat, lon=dataset_test.lon)
    ds = ds.assign_coords(time=np.arange(ntime))
    save_nc = os.path.join(save_dir, "samples.nc")
    ds.to_netcdf(save_nc)
    print(f"Saved netCDF predictions to {save_nc}")

    # ---------------------------
    # 保存灰度图：将 HR (真实 fine) 和 SR (预测 fine) 分别保存到对应文件夹
    total_psnr, total_ssim, total_rmse = 0, 0, 0
    total_PC, total_PO, total_FAR = 0, 0, 0
    total_POD, total_HSS = 0, 0
    num_samples = 0

    lr_dir = os.path.join(save_dir, "lr")
    hr_dir = os.path.join(save_dir, "HR")
    sr_dir = os.path.join(save_dir, "SR")
    os.makedirs(lr_dir, exist_ok=True)
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(sr_dir, exist_ok=True)

    for i in tqdm(range(ntime), desc="Evaluating samples"):
        pred_img = pred_array[i]
        target_img = fine_array[i]
        psnr_value, ssim_value, rmse_value, PC, PO, FAR, POD, HSS = evaluate_single_sample(pred_img, target_img,
                                                                                           threshold=0.1)
        print(f"Sample {i + 1}: PSNR={psnr_value:.2f}, SSIM={ssim_value:.4f}, RMSE={rmse_value:.4f}, "
              f"PC={PC:.2f}%, PO={PO:.2f}%, FAR={FAR:.2f}%,POD={POD:.2f}%, HSS={HSS:.2f}")
        total_psnr += psnr_value
        total_ssim += ssim_value
        total_rmse += rmse_value
        total_PC += PC
        total_PO += PO
        total_FAR += FAR
        total_POD += POD
        total_HSS += HSS
        num_samples += 1

        # 使用 _normalize_to_8bit 对每个图像做 min-max scaling 映射到 [0,255]
        hr_img = _normalize_to_8bit(fine_array[i])
        sr_img = _normalize_to_8bit(pred_array[i])
        lr_img = _normalize_to_8bit(coarse_array[i])
        hr_path = os.path.join(hr_dir, f"HR_{i + 1:04d}.png")
        sr_path = os.path.join(sr_dir, f"SR_{i + 1:04d}.png")
        lr_path = os.path.join(lr_dir, f"LR_{i + 1:04d}.png")
        cv2.imwrite(hr_path, hr_img)
        cv2.imwrite(sr_path, sr_img)
        cv2.imwrite(lr_path, lr_img)

    print(f"\nAverage metrics over {num_samples} samples:")
    print(
        f"PSNR: {total_psnr / num_samples:.2f}, SSIM: {total_ssim / num_samples:.4f}, RMSE: {total_rmse / num_samples:.4f}")
    print(f"PC: {total_PC / num_samples:.2f}%, PO: {total_PO / num_samples:.2f}%, FAR: {total_FAR / num_samples:.2f}%")
    print(f"POD:{total_POD / num_samples:.2f}%, HSS: {total_HSS / num_samples:.2f}%")
    print(f"Saved HR images to {hr_dir} and SR images to {sr_dir}")

