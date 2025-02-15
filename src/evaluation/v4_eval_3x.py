import os
import sys

import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import xarray as xr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
import numpy as np

from src.dataset.v1_dataset import PrecipitationDataset
from src.dataset.v4_dataset import PrecipitationDatasetV4
from src.models.srcnn import SRCNN
import heapq
import torch
import torch.nn.functional as F

def calculate_metrics(pred, target, threshold=0.1):
    """
    计算晴雨准确率 (PC)、漏报率 (PO)、空报率 (FAR)
    """
    pred_binary = (pred > threshold).astype(int)
    target_binary = (target > threshold).astype(int)

    NA = ((pred_binary == 1) & (target_binary == 1)).sum()  # 正确预报降雨
    NB = ((pred_binary == 1) & (target_binary == 0)).sum()  # 空报
    NC = ((pred_binary == 0) & (target_binary == 1)).sum()  # 漏报
    ND = ((pred_binary == 0) & (target_binary == 0)).sum()  # 正确无降雨

    PC = (NA + ND) / (NA + NB + NC + ND) * 100 if (NA + NB + NC + ND) > 0 else 0
    PO = NC / (NA + NC) * 100 if (NA + NC) > 0 else 0
    FAR = NB / (NA + NB) * 100 if (NA + NB) > 0 else 0

    return PC, PO, FAR


def downsample(image, size):
    """
    对图像的高度和宽度进行 bicubic 缩放。

    参数:
    - image: 输入张量，形状为 (N, C, H, W)
    - size: 目标大小 (oH, oW)，表示输出的高度和宽度

    返回:
    - 降采样后的张量，形状为 (N, C, oH, oW)
    """
    # 检查输入是否为 4D 张量
    if len(image.shape) != 4:
        raise ValueError(f"输入张量必须为 4 维 (N, C, H, W)，但得到了形状为 {image.shape} 的张量")

    # 使用 bicubic 插值对高度和宽度进行缩放
    downsampled = F.interpolate(
        image, size=size, mode='bicubic', align_corners=False
    )

    return downsampled


def evaluate_3x(test_dir, model_paths, upscale_factor, output_dir, device):
    """
    3 层模型堆叠的评估函数。
    """

    if not os.path.exists(test_dir):
        print(f"Error: The directory {test_dir} does not exist.")
        sys.exit(1)


    # 数据加载
    test_dataset = PrecipitationDatasetV4(test_dir, (224, 224), (224, 224), upscale_factor=upscale_factor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 加载 3 层模型
    models = [SRCNN(input_channels=1).to(device) for _ in range(3)]
    for i, model_path in enumerate(model_paths):
        models[i].load_state_dict(torch.load(model_path, map_location=device))  # 使用 map_location 指定设备
        models[i].to(device)  # 将模型移动到 CPU
        models[i].eval()

    # 创建结果保存目录
    sr_output_dir = os.path.join(output_dir, "results", "SR")
    hr_output_dir = os.path.join(output_dir, "results", "HR")
    os.makedirs(sr_output_dir, exist_ok=True)
    os.makedirs(hr_output_dir, exist_ok=True)

    total_psnr, total_ssim, total_rmse, total_pc, total_po, total_far = 0, 0, 0, 0, 0, 0
    count = 0
    min_rmse_heap = []

    print("\n开始评估 3 层堆叠模型...")
    with torch.no_grad():
        for idx, (lr, hr) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
            hr = hr.to(device)  # 由于Dataset设置的原因，不取出 lr 图像，由 hr 图像进行下采样

            # 将 HR 图像进行下采样到
            lr = downsample(hr, size=(28, 28))  # 下采样到 28x28

            # 逐层通过模型生成 SR 图像
            lr = F.interpolate(lr, size=(56, 56), mode='bilinear')  # 28x28 -> 56x56
            sr = models[0](lr)  # 模型第一层训练

            lr = sr
            lr = F.interpolate(lr, size=(112, 112), mode='bilinear')    # 56x56 -> 112x112
            sr = models[1](lr)

            lr = sr
            lr = F.interpolate(lr, size=(224, 224), mode='bilinear')    # 112x112 -> 224x224
            sr = models[2](lr)

            # 转为 numpy 格式用于指标计算
            sr_np = sr.cpu().numpy().squeeze()
            hr_np = hr.cpu().numpy().squeeze()

            # 计算指标
            psnr_value = calculate_psnr(sr_np, hr_np, data_range=hr_np.max() - hr_np.min())
            ssim_value = compare_ssim(sr_np, hr_np, data_range=hr_np.max() - hr_np.min())
            rmse_value = np.sqrt(np.mean((sr_np - hr_np) ** 2))
            pc, po, far = calculate_metrics(sr_np, hr_np)

            # Update min RMSE heap
            heapq.heappush(min_rmse_heap, (rmse_value, f"prediction_{idx + 1}.nc"))
            if len(min_rmse_heap) > 5:
                heapq.heappop(min_rmse_heap)

            # 累加指标
            total_psnr += psnr_value
            total_ssim += ssim_value
            total_rmse += rmse_value
            total_pc += pc
            total_po += po
            total_far += far
            count += 1

            # ============= 将 SR 和 HR 以图片形式保存 =============
            # 构建输出文件名
            sr_img_path = os.path.join(sr_output_dir, f"prediction_{idx + 1}.png")
            hr_img_path = os.path.join(hr_output_dir, f"target_{idx + 1}.png")

            # SR
            sr_img = _normalize_to_8bit(sr_np)
            cv2.imwrite(sr_img_path, sr_img)

            # HR
            hr_img = _normalize_to_8bit(hr_np)
            cv2.imwrite(hr_img_path, hr_img)

    # 计算平均指标
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_rmse = total_rmse / count
    avg_pc = total_pc / count
    avg_po = total_po / count
    avg_far = total_far / count

    # Get the 5 images with the smallest RMSE
    min_rmse_images = sorted(min_rmse_heap)

    print(f"\n评估完成！")
    print(f"平均 PSNR: {avg_psnr:.2f}, 平均 SSIM: {avg_ssim:.4f}, 平均 RMSE: {avg_rmse:.4f}")
    print(f"平均 PC: {avg_pc:.2f}%, 平均 PO: {avg_po:.2f}%, 平均 FAR: {avg_far:.2f}%")
    print(f"Top 5 images with smallest RMSE:")
    for rmse, image_name in min_rmse_images:
        print(f"Image: {image_name}, RMSE: {rmse:.4f}")

    return avg_psnr, avg_ssim, avg_rmse, avg_pc, avg_po, avg_far, min_rmse_images


def _normalize_to_8bit(array):
    """
    将浮点numpy数组 min-max 归一化到 [0, 255] 并转为 uint8。
    若整幅图所有值相同，则输出全 0。
    """
    arr_min, arr_max = array.min(), array.max()
    if np.isclose(arr_min, arr_max):
        # 全部像素相同
        return np.zeros_like(array, dtype=np.uint8)
    norm = (array - arr_min) / (arr_max - arr_min)  # [0,1]
    norm_255 = (norm * 255).astype(np.uint8)
    return norm_255

if __name__ == "__main__":
    # 配置
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    TEST_DIR = "../../processed_datasets/new_test"
    MODEL_PATHS = [
        "../../models/v4/srcnn_layer1.pth",
        "../../models/v4/srcnn_layer2.pth",
        "../../models/v4/srcnn_layer3.pth"
    ]
    UPSCALE_FACTOR = 2
    OUTPUT_DIR = "../../predictions/v4/"

    # 设备
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

    # 评估 3 层堆叠模型
    evaluate_3x(TEST_DIR, MODEL_PATHS, UPSCALE_FACTOR, OUTPUT_DIR, DEVICE)