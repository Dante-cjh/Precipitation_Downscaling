import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import xarray as xr
from pytorch_lightning import LightningDataModule


class RobustPrecipitationZScoreDataset(Dataset):
    """
    针对鲁棒性研究设计的降水超分数据集：
      - 每个样本存放在一个 .nc 文件（无时间信息），图像尺寸均为 224×224；
      - 文件中包含 7 个变量：acpcp, r2, t, u10, v10, lsm, z；
      - 对 acpcp：先生成 8 倍采样的低分辨率版本（下采样至 28×28 后再上采样至 224×224），
        计算残差 residual = fine - coarse；
      - 所有变量均采用 Z-Score 标准化，其中 acpcp 的 fine 与 coarse 分别使用其自身的均值和标准差，
        而残差使用另一组均值和标准差；
      - 此外，对 acpcp 数据进行异常值处理：保证其数值在 [0, 999] 之间；
      - 模型输入为：标准化后的 acpcp coarse（仅1通道），
        模型目标为 acpcp 的残差。
    """

    def __init__(self, data_dir, input_size=(28, 28), target_size=(224, 224), normalize=True,
                 norm_means=None, norm_stds=None, residual_mean=None, residual_std=None):
        """
        Args:
            data_dir (str): 存放 .nc 样本文件的目录，每个文件为一个样本。
            input_size (tuple): 下采样尺寸，此处应为 (28,28)。
            target_size (tuple): 原始样本尺寸，此处为 (224,224)。
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

        # 需要使用的变量列表
        self.variables = ["acpcp"]

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

        self.inverse_normalize_residual = lambda residual_norm: ((residual_norm * self.residual_std) +
                                                                 self.residual_mean)

        # 预先处理并缓存所有样本
        self.samples = []
        for file in self.file_list:
            sample = self._process_file(file)
            self.samples.append(sample)

    def _process_file(self, file_path):
        ds = xr.open_dataset(file_path)
        data = {}
        # 读取并处理变量
        for var in self.variables:
            arr = ds[var].values
            arr = np.nan_to_num(arr, nan=0.0)
            if var == "acpcp":
                arr = np.clip(arr, 0, 999)
            data[var] = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        ds.close()

        # 处理acpcp生成coarse和残差
        fine_acpcp = data["acpcp"]
        fine_unsq = fine_acpcp.unsqueeze(0)
        downsampled = F.interpolate(fine_unsq, self.input_size, mode="bicubic", align_corners=False)
        coarse_acpcp = F.interpolate(downsampled, self.target_size, mode="bicubic", align_corners=False).squeeze(0)
        residual_acpcp = fine_acpcp - coarse_acpcp

        # 标准化
        if self.normalize:
            coarse_norm = self._zscore(coarse_acpcp, self.norm_means["acpcp"], self.norm_stds["acpcp"])
            residual_norm = self._zscore(residual_acpcp, self.residual_mean, self.residual_std)
        else:
            coarse_norm = coarse_acpcp
            residual_norm = residual_acpcp


        return {
            "inputs": coarse_norm,
            "targets": residual_norm,
            "fine_acpcp": fine_acpcp,
            "coarse_acpcp": coarse_acpcp
        }


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
        return self.samples[idx]

    def residual_to_fine_image(self, residual, coarse_image):
        return coarse_image + self.inverse_normalize_residual(residual)


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
        self.train_dataset = RobustPrecipitationZScoreDataset(self.train_dir, normalize=True,
                                                              norm_stds=self.norm_stds, norm_means=self.norm_means,
                                                              residual_mean=self.residual_mean,
                                                              residual_std=self.residual_std)
        self.val_dataset = RobustPrecipitationZScoreDataset(self.val_dir, normalize=True,
                                                            norm_stds=self.norm_stds, norm_means=self.norm_means,
                                                            residual_mean=self.residual_mean,
                                                            residual_std=self.residual_std)
        self.test_dataset = RobustPrecipitationZScoreDataset(self.test_dir, normalize=True,
                                                             norm_stds=self.norm_stds, norm_means=self.norm_means,
                                                             residual_mean=self.residual_mean,
                                                             residual_std=self.residual_std)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=0)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=0)
