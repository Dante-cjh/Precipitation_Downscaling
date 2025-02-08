import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import xarray as xr
from pytorch_lightning import LightningDataModule


class PrecipitationDataset(Dataset):
    def __init__(self, hr_path, input_size, target_size, normalize=True):
        self.hr_files = sorted([os.path.join(hr_path, f) for f in os.listdir(hr_path) if f.endswith('.nc')])
        self.input_size = input_size
        self.target_size = target_size
        self.normalize = normalize

        # 初始化 min-max 值
        self.hr_min, self.hr_max = None, None
        if normalize:
            self._calculate_min_max()

    def _calculate_min_max(self):
        all_hr = []
        for hr_file in self.hr_files:
            ds = xr.open_dataset(hr_file)
            if 'acpcp' not in ds:
                continue
            hr_data = ds['acpcp'].values
            all_hr.append(hr_data)
        all_hr = np.concatenate(all_hr, axis=None)
        self.hr_min = all_hr.min()
        self.hr_max = all_hr.max()

    def normalize_data(self, data):
        return (data - self.hr_min) / (self.hr_max - self.hr_min)

    def denormalize_data(self, data):
        return data * (self.hr_max - self.hr_min) + self.hr_min

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        ds = xr.open_dataset(self.hr_files[idx])
        if 'acpcp' not in ds:
            raise ValueError(f"文件 {self.hr_files[idx]} 缺少 `acpcp` 字段")
        hr = ds['acpcp'].values
        hr = np.nan_to_num(hr, nan=0.0)
        if self.normalize:
            hr = self.normalize_data(hr)
        hr = torch.tensor(hr, dtype=torch.float32).unsqueeze(0)
        lr = F.interpolate(hr.unsqueeze(0), size=self.input_size, mode="bicubic", align_corners=False).squeeze(0)
        return lr, hr


class PrecipitationDataModule(LightningDataModule):
    def __init__(self, train_path, val_path, test_path,input_size, target_size, batch_size, normalize=True):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.input_size = input_size
        self.target_size = target_size
        self.batch_size = batch_size
        self.normalize = normalize

    def setup(self, stage=None):
        self.train_dataset = PrecipitationDataset(self.train_path, self.input_size, self.target_size, self.normalize)
        self.val_dataset = PrecipitationDataset(self.val_path, self.input_size, self.target_size, self.normalize)
        self.test_dataset = PrecipitationDataset(self.test_path, self.input_size, self.target_size, self.normalize)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)

