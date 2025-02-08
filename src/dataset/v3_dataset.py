import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

# ===== 可根据实际需求自行配置的超参数 ===== #
# 分位数裁剪区间（例如去除极端值，可调大或关闭）
CLIP_P1 = 1.0
CLIP_P99 = 99.0

# 是否对降水做 log(1 + x) 变换
APPLY_LOG_TRANSFORM = True

# 对处理后的降水是否做 min-max 归一化
APPLY_MINMAX_PRECIP = True

# 对海拔 z 是否做 min-max 归一化
APPLY_MINMAX_ALT = True

# 也可以针对海拔使用 z-score 标准化(示例):
# APPLY_ZSCORE_ALT = True  # 若开启则需在下方函数中实现

def percentile_clip(data, p1, p99):
    """
    对 numpy 数组/张量 data 进行分位数裁剪，将区间外的值裁到 [p1, p99]。
    注意：若数据量很大，请考虑在全局统计后再统一裁剪，以减少随机波动和计算量。
    """
    low_val = np.percentile(data, p1)
    high_val = np.percentile(data, p99)
    data = np.clip(data, low_val, high_val)
    return data

def transform_precip(precip_np):
    """
    针对降水(acpcp)的预处理示例:
    1) 分位数裁剪
    2) 对数变换 (可选)
    3) 再次裁剪 (可选)
    4) min-max 归一化 (可选)
    返回 numpy 数组
    """
    # 1) 分位数裁剪
    precip_np = percentile_clip(precip_np, CLIP_P1, CLIP_P99)

    # 2) 对数变换
    if APPLY_LOG_TRANSFORM:
        # 注意：只有非负降水才可做 log(1 + x)
        # 若数据有负值需先挪动
        precip_np = np.log1p(precip_np)  # log(1 + x)

        # 可选：再次分位数裁剪，防止 log 后极端情况
        precip_np = percentile_clip(precip_np, CLIP_P1, CLIP_P99)

    # 3) min-max 归一化到 [0,1]
    if APPLY_MINMAX_PRECIP:
        d_min, d_max = precip_np.min(), precip_np.max()
        if np.isclose(d_min, d_max):
            precip_np[:] = 0.0
        else:
            precip_np = (precip_np - d_min) / (d_max - d_min)

    return precip_np

def transform_altitude(alt_np):
    """
    针对海拔(z)的预处理示例:
    1) 可分位数裁剪(若海拔数据极端异常较多)
    2) 可选 min-max 归一化 或 z-score 标准化
    返回 numpy 数组
    """
    # 如果觉得海拔不需要裁剪，可注释掉
    # alt_np = percentile_clip(alt_np, CLIP_P1, CLIP_P99)

    # 若使用 min-max 归一化:
    if APPLY_MINMAX_ALT:
        a_min, a_max = alt_np.min(), alt_np.max()
        if np.isclose(a_min, a_max):
            alt_np[:] = 0.0
        else:
            alt_np = (alt_np - a_min) / (a_max - a_min)

    # 如果要使用 z-score (均值0，方差1)，可替换上面 min-max：
    # if APPLY_ZSCORE_ALT:
    #     mean_val = alt_np.mean()
    #     std_val = alt_np.std()
    #     if np.isclose(std_val, 0.0):
    #         alt_np[:] = 0.0
    #     else:
    #         alt_np = (alt_np - mean_val) / std_val

    return alt_np


class PrecipitationDatasetV3(Dataset):
    """
    用于堆叠超分训练的数据集类，支持动态加载和逐层生成。
    读取降水(acpcp)和海拔(z)，对二者分别做预处理，再组装成低分辨率与高分辨率对。
    """
    def __init__(self, data_dir, input_size, target_size, upscale_factor):
        """
        初始化数据集。

        参数:
        - data_dir: 数据目录，里面是多个 .nc 文件
        - input_size: 低分辨率输入图像的大小 (宽, 高)
        - target_size: 高分辨率目标图像的大小 (宽, 高)
        - upscale_factor: 放大倍数 (如果需要)
        """
        self.file_paths = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nc')
        ]
        self.input_size = input_size
        self.target_size = target_size
        self.upscale_factor = upscale_factor

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        动态加载并返回一对输入(lr_input)和目标(hr_target)图像。
        这里将 acpcp 视为网络要 super-res 的主要通道；z 则可拼接到输入或输出。
        
        返回:
        - lr_input:  shape = [2, H, W] (示例：第0通道是低分辨率acpcp放大后的结果，第1通道是z)
        - hr_target: shape = [1, H, W] (示例：高分辨率的acpcp)
        """
        # === 1) 读取单个文件 ===
        ds = xr.open_dataset(self.file_paths[idx])
        if 'acpcp' not in ds or 'z' not in ds:
            raise ValueError(f"文件 {self.file_paths[idx]} 缺少 `acpcp` 或者 `z` 字段")

        # (1) 提取并处理降水 acpcp
        hr_image_acpcp = ds['acpcp'].values  # numpy array
        hr_image_acpcp = np.nan_to_num(hr_image_acpcp, nan=0.0)

        # (2) 提取并处理海拔 z
        hr_image_z = ds['z'].values  # numpy array
        hr_image_z = np.nan_to_num(hr_image_z, nan=0.0)

        ds.close()  # 读取完后记得关闭

        # === 2) 对 acpcp / z 分别做预处理 ===
        hr_image_acpcp = transform_precip(hr_image_acpcp)  # shape = (H, W) or (time, H, W)...
        hr_image_z = transform_altitude(hr_image_z)

        # 如果数据有多维度(如 time, level, lat, lon)，
        # 这里仅示例取 2D：若需要多帧时序/多层高度，需自行扩展或 isel(...)
        # 假设 shape=(H, W)

        # === 3) 转为 torch.Tensor + 维度处理 ===
        # acpcp是主要做超分辨的变量 => 在网络里作为要预测的目标
        hr_image_acpcp = torch.tensor(hr_image_acpcp, dtype=torch.float32).unsqueeze(0)  # [1,H,W]
        hr_image_z = torch.tensor(hr_image_z, dtype=torch.float32)        # [1,H,W]

        # === 4) 生成高分辨率目标 (hr_target) ===
        # 你原代码使用BICUBIC插值将 "hr_image_acpcp" 强制resize到 target_size
        # 这里我们假设 "hr_image_acpcp" 本身就是高分辨率，如果分辨率不匹配，可按需做resize:
        hr_target_acpcp = TF.resize(
            hr_image_acpcp,
            size=self.target_size,
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True
        )  # => shape [1, target_H, target_W]

        # === 5) 做低分辨率模拟 + 再放大 ===
        #   a) 先将高分辨率 acpcp downscale 到 input_size
        lr_downscaled_acpcp = TF.resize(
            hr_image_acpcp,
            size=self.input_size,
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True
        )

        #   b) 再将downscale后的 acpcp用BILINEAR插值回到 target_size (SRCNN常见做法)
        lr_up_acpcp = TF.resize(
            lr_downscaled_acpcp,
            size=self.target_size,
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True
        )  # => [1, target_H, target_W]

        # === 6) 将海拔 z 也 resize 到 target_size (方便拼接)
        hr_target_z = TF.resize(
            hr_image_z,
            size=self.target_size,
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True
        )  # => [1, target_H, target_W]

        # === 7) 组合：把 acpcp(低分辨放大版) 与 海拔z 拼接到一起作为网络输入 ===
        # 例如 shape = [2, target_H, target_W], 第0通道是 LR放大的acpcp, 第1通道是z
        lr_input = torch.cat((lr_up_acpcp, hr_target_z), dim=0)

        # === 8) 输出 ===
        # lr_input: [2, target_H, target_W]
        # hr_target: [1, target_H, target_W] (只有acpcp)
        return lr_input, hr_target_acpcp