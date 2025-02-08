import os
import numpy as np
import xarray as xr
from numpy.random import geometric
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import torch

class PrecipitationDatasetV2(Dataset):
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
        if 'acpcp' and 'z' not in ds:
            raise ValueError(f"文件 {self.file_paths[idx]} 缺少 `acpcp` 或者 `z` 字段")

        # 提取高分辨率图像
        hr_image_acpcp = ds['acpcp'].values
        hr_image_z = ds['z'].values
        hr_image_acpcp = np.nan_to_num(hr_image_acpcp)  # 处理 NaN 为 0
        hr_image_z = np.nan_to_num(hr_image_z)

        # 将数据转换为 Tensor
        hr_image_acpcp = torch.tensor(hr_image_acpcp, dtype=torch.float32).unsqueeze(0)
        hr_image_z = torch.tensor(hr_image_z, dtype=torch.float32)

        # 生成目标高分辨率图像
        hr_target = TF.resize(hr_image_acpcp, size=self.target_size,
                              interpolation=TF.InterpolationMode.BICUBIC, antialias=True)

        hr_target_z = TF.resize(hr_image_z, size=self.target_size,
                              interpolation=TF.InterpolationMode.BICUBIC, antialias=True)

        # 模拟低分辨率输入
        lr_downscaled_acpcp = TF.resize(hr_image_acpcp, size=self.input_size,
                                        interpolation=TF.InterpolationMode.BICUBIC, antialias=True)

        # SRCNN是先降采样再放大，因此这里使用双线性插值
        lr_input = TF.resize(lr_downscaled_acpcp, size=self.target_size,
                             interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

        # 将hr_target_gh添加到lr_input中
        lr_input = torch.cat((lr_input, hr_target_z), dim=0)

        return lr_input, hr_target