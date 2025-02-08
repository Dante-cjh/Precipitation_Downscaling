import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.v4_dataset import PrecipitationDatasetV4
from src.models.srcnn import SRCNN
import xarray as xr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F

# PSNR 计算函数
def calculate_psnr(pred, target, data_range=None):
    """
    使用 skimage 的 psnr 函数计算 PSNR。

    参数:
    - pred: 预测图像（numpy.ndarray 或 torch.Tensor）
    - target: 目标图像（numpy.ndarray 或 torch.Tensor）
    - data_range: 图像的动态范围（通常为 1.0 或 255）

    返回:
    - PSNR 值（float）
    """
    # 如果输入是 torch.Tensor，将其转换为 numpy.ndarray
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # 确保数据维度匹配
    if pred.shape != target.shape:
        raise ValueError("预测图像和目标图像的形状必须一致")

    # 如果未指定 data_range，动态计算
    if data_range is None:
        data_range = target.max() - target.min()

    # 计算 PSNR
    return psnr(target, pred, data_range=data_range)


def calculate_metrics(pred, target, threshold=0.1):
    """
    计算晴雨准确率 (PC)、漏报率 (PO)、空报率 (FAR)

    参数:
    - pred: 预测值（numpy.ndarray 或 torch.Tensor）
    - target: 真值（numpy.ndarray 或 torch.Tensor）
    - threshold: 降雨的判定阈值

    返回:
    - PC: 晴雨准确率
    - PO: 漏报率
    - FAR: 空报率
    """
    # 转换为 numpy 数组
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    # 二值化
    pred_binary = (pred > threshold).astype(int)
    target_binary = (target > threshold).astype(int)

    # 计算分类数量
    NA = ((pred_binary == 1) & (target_binary == 1)).sum()  # 正确预报降雨
    NB = ((pred_binary == 1) & (target_binary == 0)).sum()  # 空报
    NC = ((pred_binary == 0) & (target_binary == 1)).sum()  # 漏报
    ND = ((pred_binary == 0) & (target_binary == 0)).sum()  # 正确无降雨

    # 避免零除错误
    PC = (NA + ND) / (NA + NB + NC + ND) * 100 if (NA + NB + NC + ND) > 0 else 0  # 晴雨准确率
    PO = NC / (NA + NC) * 100 if (NA + NC) > 0 else 0  # 漏报率
    FAR = NB / (NA + NB) * 100 if (NA + NB) > 0 else 0  # 空报率

    return PC, PO, FAR

# 测试函数
def evaluate_model(test_dir, input_size, target_size, model_path, upscale_factor, output_dir, device):
    # 数据加载
    test_dataset = PrecipitationDatasetV4(test_dir, input_size, target_size, upscale_factor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 加载模型
    model = SRCNN(input_channels=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 指标
    total_psnr, total_ssim, total_pc, total_po, total_far = 0, 0, 0, 0, 0
    count = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating", unit="batch")
        for idx, (lr, hr) in enumerate(progress_bar):
            lr, hr = lr.to(device), hr.to(device)
            outputs = model(lr).cpu().numpy().squeeze()
            hr = hr.cpu().numpy().squeeze()

            # 计算指标
            psnr = calculate_psnr(outputs, hr)
            ssim = compare_ssim(hr, outputs, data_range=hr.max() - hr.min())
            pc, po, far = calculate_metrics(outputs, hr)

            total_psnr += psnr
            total_ssim += ssim
            total_pc += pc
            total_po += po
            total_far += far
            count += 1

            # 获取测试文件名
            test_file_name = os.listdir(test_dir)[idx].split('.')[0]

            # 保存预测结果到 `acpcp` 字段，并包含 `latitude` 和 `longitude`
            out_path = os.path.join(output_dir, f"prediction_{test_file_name}.nc")

            # 从对应的测试文件中读取 latitude 和 longitude
            test_file_path = os.path.join(test_dir, os.listdir(test_dir)[idx])  # 获取对应文件路径
            ds = xr.open_dataset(test_file_path)  # 打开测试文件

            # 提取 latitude 和 longitude
            latitude = ds['latitude'].values if 'latitude' in ds else None
            longitude = ds['longitude'].values if 'longitude' in ds else None

            # 插值：将 outputs 调整到与 latitude 和 longitude 对应的大小
            outputs_resized = F.interpolate(
                torch.tensor(outputs).unsqueeze(0).unsqueeze(0),  # 添加批量和通道维度
                size=(len(latitude), len(longitude)),  # 目标大小（224 x 224）
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()  # 去掉多余维度，并转换为 numpy

            # 确保维度一致
            if outputs_resized.shape != (len(latitude), len(longitude)):
                raise ValueError(
                    f"插值后的预测结果维度仍然不匹配: {outputs_resized.shape} != {(len(latitude), len(longitude))}")

            # 创建包含预测结果的 Dataset
            pred_ds = xr.Dataset(
                {
                    'acpcp': (('latitude', 'longitude'), outputs_resized)  # 使用插值后的结果
                },
                coords={
                    'latitude': latitude,
                    'longitude': longitude
                }
            )

            # 保存到 NetCDF 文件
            pred_ds.to_netcdf(out_path)

    # 平均指标
    print(f"平均 PSNR: {total_psnr / count:.2f}, 平均 SSIM: {total_ssim / count:.2f}")
    print(f"平均 PC: {total_pc / count:.2f}%, 平均 PO: {total_po / count:.2f}%, 平均 FAR: {total_far / count:.2f}%")