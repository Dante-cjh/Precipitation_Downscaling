import os
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import torch

def downsample(image, target_size):
    """
    对图像进行降采样。

    参数:
    - image: 输入图像 (Tensor)
    - target_size: 目标大小 (宽，高)

    返回:
    - 降采样后的图像
    """
    return TF.resize(image, size=target_size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True)


class PrecipitationDataset(Dataset):
    """
    用于堆叠超分训练的数据集类，支持动态加载和逐层生成。
    """

    def __init__(self, data_dir, input_size, target_size, upscale_factor):
        """
        初始化数据集。

        参数:
        - data_dir: 数据目录
        - input_size: 低分辨率输入图像的大小 (宽, 高)
        - target_size: 高分辨率目标图像的大小 (宽, 高)
        - upscale_factor: 放大倍数
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
        动态加载并返回一对输入和目标图像。

        返回:
        - lr_input: 低分辨率输入图像
        - hr_target: 高分辨率目标图像
        """
        # 加载单个文件
        ds = xr.open_dataset(self.file_paths[idx])
        if 'acpcp' not in ds:
            raise ValueError(f"文件 {self.file_paths[idx]} 缺少 `acpcp` 字段")

        # 提取高分辨率图像
        hr_image = ds['acpcp'].values
        hr_image = np.nan_to_num(hr_image)  # 处理 NaN 为 0

        # 将数据转换为 Tensor
        hr_image = torch.tensor(hr_image, dtype=torch.float32).unsqueeze(0)

        # 生成目标高分辨率图像
        hr_target = TF.resize(hr_image, size=self.target_size,
                              interpolation=TF.InterpolationMode.BICUBIC, antialias=True)

        # 模拟低分辨率输入
        lr_downscaled = TF.resize(hr_target,
                                  size=(self.target_size[0] // self.upscale_factor,
                                        self.target_size[1] // self.upscale_factor),
                                  interpolation=TF.InterpolationMode.BICUBIC, antialias=True)

        # SRCNN是先降采样再放大，因此这里使用双线性插值
        lr_input = TF.resize(lr_downscaled, size=self.target_size,
                             interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

        return lr_input, hr_target