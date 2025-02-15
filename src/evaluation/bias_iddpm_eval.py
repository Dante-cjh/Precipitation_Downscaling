import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import pywt

from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    sr_create_model_and_diffusion,
    sr_model_and_diffusion_defaults
)

from pytorch_lightning import seed_everything
from improved_diffusion.ddpm import DDPM  # 你的DDPM LightningModule
from src.dataset.bias_iddpm_dataset import PrecipitationDataset

# ---------------------------------------
# =============== 指标计算 ==============
# ---------------------------------------
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr

def calculate_metrics(pred, target, threshold=0.1):
    """
    计算二值指标：PC, PO, FAR
    threshold: 判断是否降水的阈值
    """
    pred_binary = (pred > threshold).astype(int)
    target_binary = (target > threshold).astype(int)
    NA = ((pred_binary == 1) & (target_binary == 1)).sum()
    NB = ((pred_binary == 1) & (target_binary == 0)).sum()
    NC = ((pred_binary == 0) & (target_binary == 1)).sum()
    ND = ((pred_binary == 0) & (target_binary == 0)).sum()
    total = NA + NB + NC + ND
    PC = (NA + ND) / total * 100 if total > 0 else 0
    sumA = NA + NC
    PO = NC / sumA * 100 if sumA > 0 else 0
    sumB = NA + NB
    FAR = NB / sumB * 100 if sumB > 0 else 0
    return PC, PO, FAR

def evaluate_single_sample(pred_img, target_img, threshold=0.1):
    """
    计算 PSNR, SSIM, RMSE, PC, PO, FAR
    data_range 动态范围使用 (target.max()-target.min())
    """
    data_range = target_img.max() - target_img.min()  # 避免固定 1.0
    if data_range < 1e-12:
        data_range = 1.0  # 防止极端情况
    psnr_value = calculate_psnr(target_img, pred_img, data_range=data_range)
    ssim_value = compare_ssim(target_img, pred_img, data_range=data_range)
    rmse_value = np.sqrt(np.mean((pred_img - target_img) ** 2))

    PC, PO, FAR = calculate_metrics(pred_img, target_img, threshold=threshold)
    return psnr_value, ssim_value, rmse_value, PC, PO, FAR

def _normalize_to_8bit(array):
    """
    将 array min-max 归一化到 [0,255] (uint8)，
    用于输出PNG灰度图
    """
    arr_min, arr_max = array.min(), array.max()
    if np.isclose(arr_min, arr_max):
        return np.zeros_like(array, dtype=np.uint8)
    norm = (array - arr_min) / (arr_max - arr_min)
    norm_255 = (norm * 255).astype(np.uint8)
    return norm_255

# ============ 关键函数：对 wavelet (4,H/2,W/2) 做逆变换回 (1,H,W) ============
def inverse_wavelet_4ch(array_4hw, wave="haar"):
    """
    对 (4, H/2, W/2) 格式的子带执行 inverse DWT2。
    array_4hw: shape = (4, h, w)
       - 其中 array_4hw[0] = LL, array_4hw[1] = LH, array_4hw[2] = HL, array_4hw[3] = HH
    返回 shape = (H, W)
    """
    LL = array_4hw[0]
    LH = array_4hw[1]
    HL = array_4hw[2]
    HH = array_4hw[3]
    # pywt.idwt2 的输入: (LL, (LH,HL,HH))
    out = pywt.idwt2((LL, (LH, HL, HH)), wave)
    return out  # shape ~ (2*h, 2*w)

def main():
    seed_everything(42)  # 固定随机种子(可选)

    # ========== 1. 路径和设备 ==========
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # large_size 和 small_size 决定 HR & LR 尺寸
    large_size = 224
    small_size = 28

    # 你的最佳模型（.ckpt）路径
    best_ckpt_path = "../../models/bias_diffusion/epoch=32-Loss=0.0000.ckpt"
    # 测试数据目录
    test_dir = "../../processed_datasets/new_test"
    # 预测结果输出目录
    save_dir = "../../predictions/bias_iddpm"
    os.makedirs(save_dir, exist_ok=True)

    # =========== 2) 构造模型 & Diffusion ===========
    in_channels = 1
    cond_channels = 7  # wavelet=True => 低分(7->7*4=28) ?

    model_conf = sr_model_and_diffusion_defaults()
    model_conf["large_size"] = large_size
    model_conf["small_size"] = small_size
    model_conf["num_channels"] = 128
    model_conf["num_res_blocks"] = 2
    model_conf["diffusion_steps"] = 4000
    model_conf["noise_schedule"] = "linear"
    model_conf["learn_sigma"] = True
    model_conf["class_cond"] = False
    model_conf["wavelet"] = True  # 如果训练时也是 True

    print("[main] Creating SR model & diffusion ...")
    from improved_diffusion.resample import create_named_schedule_sampler
    model, diffusion = sr_create_model_and_diffusion(
        in_channels=in_channels,
        cond_channels=cond_channels,
        **model_conf,
    )
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    model.to(device)

    # ========== 3. 加载DDPM权重 ==========
    from improved_diffusion.ddpm import DDPM
    lr = 1e-4
    ema_rate = "0.9999"
    weight_decay = 0.0

    print(f"Loading best model from: {best_ckpt_path}")
    ddpm_module = DDPM.load_from_checkpoint(
        checkpoint_path=best_ckpt_path,
        model=model,
        diffusion=diffusion,
        lr=lr,
        ema_rate=ema_rate,
        schedule_sampler=schedule_sampler,
        weight_decay=weight_decay,
        model_dir=save_dir
    )
    ddpm_module.to(device)
    ddpm_module.eval()

    # ========== 4. 构造测试集，wavelet=True ==========
    test_dataset = PrecipitationDataset(
        data_dir=test_dir,
        input_size=(small_size, small_size),
        target_size=(large_size, large_size),
        use_condition=True,
        wavelet=True,   # <-- 这里保持true
        gamma=True
    )
    ntime = len(test_dataset)
    print(f"Test samples: {ntime}")

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    # ========== 5. 预测 & 逆小波还原 ==========
    # wavelet=True => hr: (4,112,112), lr: (28,112,112) ? or (B,4,112,112)
    # p_sample_loop => sample 也是 wavelet(4,112,112)? 取决于 SuperResModel 是否在 forward 里 wavelet
    # => 需要 inverse wavelet -> (1,224,224)
    # 统计指标
    total_psnr, total_ssim, total_rmse = 0, 0, 0
    total_PC, total_PO, total_FAR = 0, 0, 0
    num_samples = 0

    pred_array = np.zeros((ntime, large_size, large_size), dtype=np.float32)
    hr_array   = np.zeros((ntime, large_size, large_size), dtype=np.float32)

    idx_offset = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inference")):
            hr = batch["hr"].to(device)  # shape: (B,4,112,112)
            lr = batch["lr"].to(device)  # shape: (B,28,112,112)
            B = hr.shape[0]

            # 走扩散采样 => sample => wavelet形状? => (B,4,112,112) ?
            sample = diffusion.p_sample_loop(
                ddpm_module.model,
                shape=(B, hr.shape[1], 112, 112),  # wavelet通道=4, spatial=112
                model_kwargs={"low_res": lr},
                clip_denoised=True
            )
            # sample shape (B,4,112,112)
            sample_np = sample.detach().cpu().numpy()
            hr_np     = hr.detach().cpu().numpy()

            # 逆小波 => (1,224,224)
            for i in range(B):
                # hr_np[i] shape (4,112,112)
                hr_inv = inverse_wavelet_4ch(hr_np[i])      # => (224,224)
                pred_inv = inverse_wavelet_4ch(sample_np[i])# => (224,224)

                hr_array[idx_offset + i] = hr_inv
                pred_array[idx_offset + i] = pred_inv
            idx_offset += B

    # 6. 计算指标 + 保存可视化
    hr_dir = os.path.join(save_dir, "HR")
    sr_dir = os.path.join(save_dir, "SR")
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(sr_dir, exist_ok=True)

    for i in tqdm(range(ntime), desc="Evaluating"):
        pred_img = pred_array[i]
        hr_img   = hr_array[i]

        psnr_val, ssim_val, rmse_val, PC, PO, FAR = evaluate_single_sample(pred_img, hr_img, threshold=0.1)
        total_psnr += psnr_val
        total_ssim += ssim_val
        total_rmse += rmse_val
        total_PC   += PC
        total_PO   += PO
        total_FAR  += FAR
        num_samples += 1

        # 保存灰度图
        hr_img_8  = _normalize_to_8bit(hr_img)
        pred_img_8= _normalize_to_8bit(pred_img)
        cv2.imwrite(os.path.join(hr_dir, f"HR_{i:04d}.png"), hr_img_8)
        cv2.imwrite(os.path.join(sr_dir, f"SR_{i:04d}.png"), pred_img_8)

    print(f"\nAverage metrics over {num_samples} samples:")
    print(f"  PSNR: {total_psnr/num_samples:.2f}, SSIM: {total_ssim/num_samples:.4f}, "
          f"RMSE: {total_rmse/num_samples:.4f}")
    print(f"  PC: {total_PC/num_samples:.2f}%, PO: {total_PO/num_samples:.2f}%, FAR: {total_FAR/num_samples:.2f}%")
    print(f"Images saved to {save_dir}/HR, {save_dir}/SR")

if __name__ == "__main__":
    main()