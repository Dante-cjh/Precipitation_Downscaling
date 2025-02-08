import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import heapq
import torch
import torch.nn.functional as F
import cv2  # 用于写图
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr

from src.dataset.v2_dataset import PrecipitationDatasetV2
from src.dataset.v3_dataset import PrecipitationDatasetV3
from src.models.srcnn import SRCNN


def calculate_metrics(pred, target, threshold=0.1):
    """
    计算晴雨准确率 (PC)、漏报率 (PO)、空报率 (FAR)
    pred、target 都是 numpy 数组
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
    image: (N, C, H, W)
    size: (oH, oW)
    """
    if len(image.shape) != 4:
        raise ValueError(f"输入张量必须为 4D (N, C, H, W)，但得到了形状为 {image.shape}")
    downsampled = F.interpolate(
        image, size=size, mode='bicubic', align_corners=False
    )
    return downsampled

def evaluate_3x_v3(test_dir, model_paths, upscale_factor, output_dir, device):
    """
    3层模型堆叠的评估函数, 输出模型预测(SR)和目标(HR)为图片
    """
    # ============== 1) 数据加载 ==============
    test_dataset = PrecipitationDatasetV3(
        data_dir=test_dir,
        input_size=(224, 224),
        target_size=(224, 224),
        upscale_factor=upscale_factor
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ============== 2) 加载 3层模型 ==============
    models = [SRCNN().to(device) for _ in range(3)]
    for i, model_path in enumerate(model_paths):
        models[i].load_state_dict(torch.load(model_path, map_location=device))
        models[i].eval()

    # ============== 3) 创建结果保存目录 (SR / HR) ==============
    sr_output_dir = os.path.join(output_dir, "results", "SR")
    hr_output_dir = os.path.join(output_dir, "results", "HR")
    os.makedirs(sr_output_dir, exist_ok=True)
    os.makedirs(hr_output_dir, exist_ok=True)

    total_psnr, total_ssim, total_rmse = 0, 0, 0
    total_pc, total_po, total_far = 0, 0, 0
    count = 0
    min_rmse_heap = []  # 用于找RMSE最小的若干样本

    print("\n开始评估 3 层堆叠模型 (v3)...")
    with torch.no_grad():
        # 用 tqdm 做进度可视化
        for idx, (lr, hr_acpcp) in tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc="Evaluating"
        ):
            # lr: [1, 2, H, W]  (channel0=低分降雨, channel1=地形)
            # hr_acpcp: [1, 1, H, W]  (仅高分降雨)

            hr_z = lr[:, 1:, :, :].to(device)   # 地形图 (HR)
            hr_acpcp = hr_acpcp.to(device)      # 真值(高分降雨)

            # ================== 第1层输入 (28x28 -> 56x56) ==================
            lr_acpcp_1 = downsample(hr_acpcp, size=(28, 28))
            lr_acpcp_1 = F.interpolate(lr_acpcp_1, size=(56, 56), mode='bilinear')
            hr_z_1 = downsample(hr_z, size=(56, 56))

            input_layer1 = torch.cat((lr_acpcp_1, hr_z_1), dim=1)
            sr_layer1 = models[0](input_layer1)

            # ================== 第2层输入 (56x56 -> 112x112) ==================
            lr_acpcp_2 = F.interpolate(sr_layer1, size=(112, 112), mode='bilinear')
            hr_z_2 = downsample(hr_z, size=(112, 112))

            input_layer2 = torch.cat((lr_acpcp_2, hr_z_2), dim=1)
            sr_layer2 = models[1](input_layer2)

            # ================== 第3层输入 (112x112 -> 224x224) ==================
            lr_acpcp_3 = F.interpolate(sr_layer2, size=(224, 224), mode='bilinear')
            hr_z_3 = downsample(hr_z, size=(224, 224))

            input_layer3 = torch.cat((lr_acpcp_3, hr_z_3), dim=1)
            sr_layer3 = models[2](input_layer3)

            # ================== 拿到最终 SR / HR numpy ==============
            sr_np = sr_layer3.cpu().numpy().squeeze()
            hr_np = hr_acpcp.cpu().numpy().squeeze()

            # ================== 计算评价指标 ==================
            # 注意 data_range=hr_np.max()-hr_np.min() 仅用于PSNR/SSIM
            psnr_value = calculate_psnr(
                sr_np, hr_np, data_range=hr_np.max() - hr_np.min() if hr_np.max() > hr_np.min() else 1
            )
            ssim_value = compare_ssim(
                sr_np, hr_np, data_range=hr_np.max() - hr_np.min() if hr_np.max() > hr_np.min() else 1
            )
            rmse_value = np.sqrt(np.mean((sr_np - hr_np) ** 2))

            pc, po, far = calculate_metrics(sr_np, hr_np)

            # 保存RMSE最小topN
            heapq.heappush(min_rmse_heap, (rmse_value, f"prediction_{idx + 1}.png"))
            if len(min_rmse_heap) > 5:
                heapq.heappop(min_rmse_heap)

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

    # ============= 计算平均指标 =============
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_rmse = total_rmse / count
    avg_pc = total_pc / count
    avg_po = total_po / count
    avg_far = total_far / count

    # 获取 RMSE 最小的 5 张图片
    min_rmse_images = sorted(min_rmse_heap)

    print(f"\n评估完成 (v3)！")
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


# ================== 以下是主调用示例 ==================
if __name__ == "__main__":
    TEST_DIR = "../../processed_datasets/new_test"
    MODEL_PATHS = [
        "../../models/v3/srcnn_layer1.pth",
        "../../models/v3/srcnn_layer2.pth",
        "../../models/v3/srcnn_layer3.pth"
    ]
    UPSCALE_FACTOR = 2
    OUTPUT_DIR = "../../predictions/v3/"

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

    evaluate_3x_v3(
        test_dir=TEST_DIR,
        model_paths=MODEL_PATHS,
        upscale_factor=UPSCALE_FACTOR,
        output_dir=OUTPUT_DIR,
        device=DEVICE
    )