import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import xarray as xr
from pytorch_lightning import LightningDataModule


class RobustPrecipitationZScoreDataset(Dataset):
    """
        增强型降水降尺度数据集，支持多模态条件输入和极端降水加权
        主要改进点：
        1. 添加降水强度掩码生成（用于加权损失）
        2. 显式分离静态/动态气象条件
        3. 支持地形自适应归一化
    """
    STATIC_VARS = ["lsm", "z"]  # 静态变量（地形相关）
    DYNAMIC_VARS = ["r2", "t", "u10", "v10"]  # 动态气象变量

    def __init__(self, data_dir, input_size=(28, 28), target_size=(224, 224), normalize=True,
                 extreme_threshold=50.0, weight_scale=3.0,
                 norm_means=None, norm_stds=None, residual_mean=None, residual_std=None):
        """
        Args:
            data_dir (str): 存放 .nc 样本文件的目录，每个文件为一个样本。
            input_size (tuple): 下采样尺寸，此处应为 (28,28)。
            target_size (tuple): 原始样本尺寸，此处为 (224,224)。
            extreme_threshold (float): 极端降水阈值（单位：mm）
            weight_scale (float): 极端降水区域的损失权重缩放因子
            normalize (bool): 是否对变量进行 Z-Score 标准化。
            norm_means (dict): 各变量的均值，键为变量名（acpcp, r2, t, u10, v10, lsm, z）。
            norm_stds  (dict): 各变量的标准差，键同上。
            residual_mean (float): acpcp 残差的均值。
            residual_std  (float): acpcp 残差的标准差。

        若未提供 norm_means/norm_stds/residual_mean/residual_std，则自动遍历数据集计算。
        """
        self.data_dir = data_dir
        self.file_list = sorted([os.path.join(data_dir, f)
                                 for f in os.listdir(data_dir) if f.endswith('.nc')])
        self.input_size = input_size
        self.target_size = target_size
        self.normalize = normalize
        self.extreme_threshold = extreme_threshold
        self.weight_scale = weight_scale

        # 需要使用的变量列表
        self.variables = ["acpcp", "r2", "t", "u10", "v10", "lsm", "z"]

        # 若未提供归一化参数，则计算所有变量和残差的均值与标准差
        if norm_means is None or norm_stds is None or residual_mean is None or residual_std is None:
            comp_means, comp_stds, comp_res_mean, comp_res_std = self._compute_normalization_params()
            self.norm_means = norm_means if norm_means is not None else comp_means
            self.norm_stds = norm_stds if norm_stds is not None else comp_stds
            self.residual_mean = residual_mean if residual_mean is not None else comp_res_mean
            self.residual_std = residual_std if residual_std is not None else comp_res_std
        else:
            self.norm_means = norm_means
            self.norm_stds = norm_stds
            self.residual_mean = residual_mean
            self.residual_std = residual_std


    def _compute_normalization_params(self):
        """
        遍历所有文件，计算各变量及 acpcp 残差的均值与标准差。
        对于 acpcp 变量，先执行异常值处理（clip 到 [0,999]）。
        """
        data_dict = {var: [] for var in self.variables}  # 存放每个变量的所有样本（展平后的一维数组）
        residual_list = []
        for file in self.file_list:
            ds = xr.open_dataset(file)
            # 处理各变量
            for var in self.variables:
                if var not in ds:
                    ds.close()
                    raise ValueError(f"文件 {file} 缺少变量 {var}")
                arr = ds[var].values  # shape: (224,224)
                arr = np.nan_to_num(arr, nan=0.0)
                if var == "acpcp":
                    # 对降水数据进行异常值处理：clip 到 [0,999]
                    arr = np.clip(arr, 0, 999)
                data_dict[var].append(arr.flatten())
            # 计算 acpcp 的残差：先生成 low-res 版本，再 residual = fine - coarse
            fine = torch.tensor(ds["acpcp"].values, dtype=torch.float32)
            fine = torch.clamp(fine, 0, 999)  # 异常值处理
            # 扩展维度为 (1,1,224,224)
            fine_unsq = fine.unsqueeze(0).unsqueeze(0)
            # 下采样到 input_size，再上采样回 target_size（保持原始网格）
            downsampled = F.interpolate(fine_unsq, size=self.input_size, mode="bicubic", align_corners=False)
            coarse = F.interpolate(downsampled, size=self.target_size, mode="bicubic", align_corners=False)
            residual = fine_unsq - coarse  # shape (1,1,224,224)
            residual_np = residual.squeeze().numpy()  # shape (224,224)
            residual_list.append(residual_np.flatten())
            ds.close()

        # 计算各变量均值和标准差
        computed_norm_means = {}
        computed_norm_stds = {}
        for var in self.variables:
            all_data = np.concatenate(data_dict[var], axis=0)
            computed_norm_means[var] = float(np.mean(all_data))
            computed_norm_stds[var] = float(np.std(all_data))
        # 计算残差均值和标准差
        all_residual = np.concatenate(residual_list, axis=0)
        computed_residual_mean = float(np.mean(all_residual))
        computed_residual_std = float(np.std(all_residual))
        return computed_norm_means, computed_norm_stds, computed_residual_mean, computed_residual_std

    def _zscore(self, tensor, mean, std):
        """对 tensor 进行 Z-Score 标准化： (x - mean) / std"""
        return (tensor - mean) / std

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        ds = xr.open_dataset(file_path)

        # 读取所有变量，均假设 shape 为 (224,224)，并转换为 tensor（添加 channel 维度）
        data = {}
        for var in self.variables:
            if var not in ds:
                ds.close()
                raise ValueError(f"文件 {file_path} 缺少变量 {var}")
            arr = ds[var].values
            arr = np.nan_to_num(arr, nan=0.0)
            if var == "acpcp":
                # 异常值处理：clip 到 [0, 999]
                arr = np.clip(arr, 0, 999)
            data[var] = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        ds.close()

        # ----- 针对 acpcp（降水）处理 -----
        # 原始高分辨率降水图像（未标准化）：shape (1,224,224)
        fine_acpcp = data["acpcp"]
        coarse_acpcp = self._generate_lowres(fine_acpcp)  # shape (1,224,224)

        # 计算残差
        residual_acpcp = fine_acpcp - coarse_acpcp

        if self.normalize:
            # 对 acpcp 的 coarse 做 Z-Score 标准化（使用计算得到的参数）
            fine_norm_acpcp = self._zscore(fine_acpcp, self.norm_means["acpcp"], self.norm_stds["acpcp"])
            coarse_norm_acpcp = self._zscore(coarse_acpcp, self.norm_means["acpcp"], self.norm_stds["acpcp"])
            # 计算残差（在原始空间），再用残差的参数进行标准化
            residual_norm = self._zscore(residual_acpcp, self.residual_mean, self.residual_std)
        else:
            fine_norm_acpcp = fine_acpcp
            coarse_norm_acpcp = coarse_acpcp
            residual_norm = residual_acpcp

        # 生成降水强度掩码
        rainfall_mask = self._generate_rainfall_mask(fine_acpcp)

        # ----- 其它变量处理（r2, t, u10, v10, lsm, z） -----
        other_vars_norm = {}
        for var in self.variables:
            if var == "acpcp":
                continue  # 已处理
            if self.normalize:
                other_vars_norm[var] = self._zscore(data[var], self.norm_means[var], self.norm_stds[var])
            else:
                other_vars_norm[var] = data[var]

            # 检查变量的 shape，如果不是 (a, a, a)，则 reshape 为 (a, a, a)
            if other_vars_norm[var].ndimension() != 3:
                other_vars_norm[var] = other_vars_norm[var].squeeze(0)

        # 多模态条件整合
        conditions = {
            'dynamic': torch.stack([other_vars_norm[var].squeeze(0) for var in self.DYNAMIC_VARS], dim=0),
            'static': torch.stack([other_vars_norm[var].squeeze(0) for var in self.STATIC_VARS], dim=0),
            'coarse_precip': coarse_norm_acpcp
        }

        # 返回的字典中包含：
        #   "inputs"       : 模型输入（低分辨率 acpcp + 全分辨率其它数据，共 7 通道）
        #   "targets"      : acpcp 残差（标准化后），shape (1,224,224)
        #   "fine_acpcp"   : 标准化后的高分辨率 acpcp（可用于可视化）
        #   "coarse_acpcp" : 标准化后的低分辨率 acpcp
        #   其它变量也一并返回（标准化后）
        sample = {
            'conditions': conditions,  # 多模态条件输入
            'target': residual_norm,  # 标准化残差
            'rainfall_mask': rainfall_mask,  # 降水强度掩码
            'metadata': {
                'fine_precip': fine_acpcp,
                'coarse_precip': coarse_acpcp,
                'file_path': file_path
            }
        }

        return sample

    def _generate_lowres(self, fine_precip):
        """改进的下采样方法，保持边缘信息"""
        fine_unsq = fine_precip.unsqueeze(0)  # [1,1,H,W]
        downsampled = F.interpolate(fine_unsq, size=self.input_size,
                                    mode='bicubic', align_corners=False)

        # 确保 downsampled 形状正确，再进行第二次下采样
        downsampled = F.interpolate(downsampled, size=self.target_size,
                                    mode='bicubic', align_corners=False)
        return downsampled.squeeze(0)


    def _generate_rainfall_mask(self, precip):
        """生成降水强度权重掩码"""
        mask = torch.ones_like(precip)
        extreme_mask = (precip >= self.extreme_threshold)
        mask[extreme_mask] = self.weight_scale
        return mask

    def residual_to_fine_image_v2(self, residual, coarse):
        return coarse + self.inverse_normalize['residual'](residual)


class RobustPrecipitationDataModule(LightningDataModule):
    def __init__(self, train_dir, val_dir, test_dir, batch_size=8,
                 norm_means=None, norm_stds=None, residual_mean=None, residual_std=None):
        """
        Args:
            train_dir, val_dir, test_dir: 分别为训练、验证和测试数据所在目录，
                每个目录内的 .nc 文件均为单个样本；
            batch_size: 每个批次大小。
        """
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.norm_means = norm_means
        self.norm_stds = norm_stds
        self.residual_mean = residual_mean
        self.residual_std = residual_std


    def setup(self, stage=None):
        # 请确保 robust_precipitation_dataset.py 文件中包含 RobustPrecipitationZScoreDataset 类
        self.train_dataset = RobustPrecipitationZScoreDataset(self.train_dir, norm_means=self.norm_means, norm_stds=self.norm_stds,
                                                              residual_mean=self.residual_mean, residual_std=self.residual_std)
        self.val_dataset = RobustPrecipitationZScoreDataset(self.val_dir, norm_means=self.norm_means, norm_stds=self.norm_stds,
                                                              residual_mean=self.residual_mean, residual_std=self.residual_std)
        self.test_dataset = RobustPrecipitationZScoreDataset(self.test_dir, norm_means=self.norm_means, norm_stds=self.norm_stds,
                                                              residual_mean=self.residual_mean, residual_std=self.residual_std)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=0,
                                           pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=0,
                                           pin_memory=True)
