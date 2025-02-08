import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from pytorch_lightning import LightningDataModule
from torchvision.transforms import functional as TF

from src.util import array3Dto2Dmat


def handle_missing_values(data, strategy="mean"):
    if strategy == "zero":
        return np.nan_to_num(data)
    elif strategy == "mean":
        return np.nan_to_num(data, nan=np.nanmean(data))


def log_transform(data):
    return np.log1p(data)  # log(1 + data)


class PrecipitationDataset(Dataset):
    def __init__(self, hr_path, input_size, target_size, normalize=True):
        self.hr_files = sorted([os.path.join(hr_path, f) for f in os.listdir(hr_path) if f.endswith('.nc')])
        self.input_size = input_size
        self.target_size = target_size
        self.normalize = normalize

        # Initialize min-max values
        self.min_max_values = {}
        if normalize:
            self._calculate_min_max()

    def _calculate_min_max(self):
        all_data = {var: [] for var in ['acpcp', 'z', 'lsm', 'r2', 't', 'u10', 'v10']}
        for hr_file in self.hr_files:
            ds = xr.open_dataset(hr_file)
            for var in all_data.keys():
                if var in ds:
                    data = ds[var].values
                    all_data[var].append(data)
        for var, data_list in all_data.items():
            all_data[var] = np.concatenate(data_list, axis=None)
            self.min_max_values[var] = (all_data[var].min(), all_data[var].max())

    def normalize_data(self, data, var_name):
        min_val, max_val = self.min_max_values[var_name]
        return (data - min_val) / (max_val - min_val)

    def denormalize_data(self, data, var_name):
        min_val, max_val = self.min_max_values[var_name]
        return data * (max_val - min_val) + min_val

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        file_path = self.hr_files[idx]

        # Load the dataset
        ds = xr.open_dataset(file_path)

        # Extract the main precipitation data
        if 'acpcp' not in ds:
            raise ValueError(f"File {file_path} is missing the `acpcp` field")
        hr_image = ds['acpcp'].values
        hr_image = handle_missing_values(hr_image)  # Replace NaN with 0

        # Normalize if required
        if self.normalize:
            hr_image = self.normalize_data(hr_image, 'acpcp')

        # Extract auxiliary variables
        aux_vars = []
        for var in ['z', 'lsm', 'r2', 't', 'u10', 'v10']:
            if var in ds:
                aux_var = ds[var].values
                aux_var = handle_missing_values(aux_var)
                if self.normalize:
                    aux_var = self.normalize_data(aux_var, var)
                aux_vars.append(aux_var)
            else:
                raise ValueError(f"File {file_path} is missing the auxiliary variable `{var}`")

        # Convert to NumPy arrays
        hr_image = np.array(hr_image, dtype=np.float32)
        aux_vars = [np.array(aux, dtype=np.float32) for aux in aux_vars]

        # Construct the high-resolution target image
        hr_target = torch.tensor(hr_image, dtype=torch.float32).unsqueeze(0)
        hr_target = TF.resize(hr_target, size=self.target_size,
                              interpolation=TF.InterpolationMode.BICUBIC, antialias=True)

        # Simulate low-resolution input
        lr_input = TF.resize(hr_target, size=self.input_size,
                             interpolation=TF.InterpolationMode.BICUBIC, antialias=True)

        # Downsample auxiliary variables
        aux_vars_resized = [TF.resize(torch.tensor(aux, dtype=torch.float32).unsqueeze(0), size=self.input_size,
                                      interpolation=TF.InterpolationMode.BICUBIC, antialias=True).squeeze(0)
                            for aux in aux_vars]
        aux_vars_resized = [aux.unsqueeze(0) if aux.ndim == 2 else aux for aux in aux_vars_resized]
        # Stack all variables to form the low-resolution input
        lr_input = torch.cat([lr_input] + aux_vars_resized, dim=0)

        # 模型设计原因，需要将hr_target转换为2D (1,224,224) -> (1,224*224)
        hr_target = array3Dto2Dmat(hr_target).squeeze(0)

        return lr_input, hr_target


class PrecipitationDataModule(LightningDataModule):
    def __init__(self, train_path, val_path, test_path, input_size, target_size, batch_size, normalize=True):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.input_size = input_size
        self.target_size = target_size
        self.batch_size = batch_size
        self.normalize = normalize
        self.num_workers = min(79, os.cpu_count())  # 确保不超过 79

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None


    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = PrecipitationDataset(self.train_path, self.input_size, self.target_size,
                                                      self.normalize)
            self.val_dataset = PrecipitationDataset(self.val_path, self.input_size, self.target_size, self.normalize)
        if stage == "test" or stage is None:
            self.test_dataset = PrecipitationDataset(self.test_path, self.input_size, self.target_size, self.normalize)


    def train_dataloader(self):
        print(f"Initializing DataLoader for train_dataset with {len(self.train_dataset)} samples.")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        print(f"Initializing DataLoader for train_dataset with {len(self.val_dataset)} samples.")
        return DataLoader(self.val_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        # 实现 test_dataloader 方法
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False)