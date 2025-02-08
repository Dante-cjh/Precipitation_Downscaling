import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class PrecipitationDatasetCGAN(Dataset):
    """
    用于堆叠超分训练的数据集类，支持动态加载和逐层生成。
    """

    def __init__(self, data_dir, input_size, target_size, upscale_factor, in_depth=1, cvars=[], log_transform=True):
        """
        初始化数据集。

        参数:
        - data_dir: 数据目录
        - input_size: 低分辨率输入图像的大小 (宽, 高)
        - target_size: 高分辨率目标图像的大小 (宽, 高)
        - upscale_factor: 放大倍数
        - in_depth: 时间序列深度
        - cvars: 辅助变量列表
        - log_transform: 是否对降水数据进行对数变换
        - test_split: 测试集划分比例
        """
        self.file_paths = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nc')
        ]
        self.input_size = input_size
        self.target_size = target_size
        self.upscale_factor = upscale_factor
        self.in_depth = in_depth
        self.cvars = cvars
        self.log_transform = log_transform


    def __len__(self):
        return len(self.file_paths) - self.in_depth + 1

    def __getitem__(self, idx):
        """
        动态加载并返回一对输入和目标图像。

        返回:
        - lr_input: 低分辨率输入图像
        - hr_target: 高分辨率目标图像
        - aux_vars: 辅助变量张量
        """
        # 提取时间序列范围
        file_range = self.file_paths[idx: idx + self.in_depth]

        # 初始化变量容器
        hr_images = []
        aux_vars = {var: [] for var in self.cvars}
        z_images = []

        # 加载时间序列内的所有文件
        for file_path in file_range:
            ds = xr.open_dataset(file_path)

            # 提取主要降水数据
            if 'acpcp' not in ds:
                raise ValueError(f"文件 {file_path} 缺少 `acpcp` 字段")
            hr_image = ds['acpcp'].values
            hr_image = np.nan_to_num(hr_image)  # 将 NaN 替换为 0

            if self.log_transform:
                hr_image = np.log1p(hr_image)  # 对数变换

            hr_images.append(hr_image)

            # 提取辅助变量
            for var in self.cvars:
                if var in ds:
                    if var == "lsm":
                        aux_var = ds[var].values[0, :, :]  # 取出时间维度第一个数据
                    else:
                        aux_var = ds[var].values
                    aux_var = np.nan_to_num(aux_var)
                    aux_vars[var].append(aux_var)
                else:
                    raise ValueError(f"文件 {file_path} 缺少辅助变量 `{var}`")

            # 提取海拔数据
            if "z" not in ds:
                raise ValueError(f"文件 {file_path} 缺少 `z` 字段")
            z = ds["z"].values
            z = np.nan_to_num(z)
            z_images.append(z)

        # 转换为 NumPy 数组
        hr_images = np.array(hr_images, dtype=np.float32)
        aux_vars = {var: np.array(aux_vars[var], dtype=np.float32) for var in self.cvars}
        z_images = np.array(z_images, dtype=np.float32)

        # 构建目标高分辨率图像
        hr_target = torch.tensor(hr_images[-1], dtype=torch.float32).unsqueeze(0)
        hr_target = TF.resize(hr_target, size=self.target_size,
                              interpolation=TF.InterpolationMode.BICUBIC, antialias=True)

        z_target = torch.tensor(z_images[-1], dtype=torch.float32)
        z_target = TF.resize(z_target, size=self.target_size,
                             interpolation=TF.InterpolationMode.BICUBIC, antialias=True)

        # 模拟低分辨率输入
        lr_input = TF.resize(hr_target,
                                  size=(self.target_size[0] // self.upscale_factor,
                                        self.target_size[1] // self.upscale_factor),
                                  interpolation=TF.InterpolationMode.BICUBIC, antialias=True)

        # 对辅助变量进行下采样
        for var in aux_vars:
            aux_vars[var] = [TF.resize(torch.tensor(aux, dtype=torch.float32).unsqueeze(0),
                                       size=(self.target_size[0] // self.upscale_factor,
                                             self.target_size[1] // self.upscale_factor),
                                       interpolation=TF.InterpolationMode.BICUBIC, antialias=True).squeeze(0)
                             for aux in aux_vars[var]]

        # 转换辅助变量为张量
        aux_vars_tensor = [torch.stack([aux.clone().detach().float() for aux in aux_vars[var]]) for var in self.cvars]

        return lr_input, aux_vars_tensor, hr_target, z_target
