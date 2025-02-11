import os
import numpy as np
import pywt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import xarray as xr
from pytorch_lightning import LightningDataModule


class PrecipitationDataset(Dataset):
    def __init__(
        self,
        data_dir,
        input_size=(28, 28),
        target_size=(224, 224),
        use_condition=True,
        wavelet=True,
        gamma=True,
    ):
        """
        Args:
            data_dir (str): 存放 .nc 文件的目录，每个文件代表一个样本
            input_size (tuple): 低分辨率下采样目标大小(默认28×28)
            target_size (tuple): 高分辨率原图大小(默认224×224)
            use_condition (bool): 是否使用其它气象变量作为条件通道
            wavelet (bool): 是否对所有通道做二维小波变换
            gamma (bool): 是否对所有通道执行 (clip / 999)**0.15 形式的 gamma 校正
        """
        super().__init__()
        self.data_dir = data_dir
        self.file_list = sorted(
            [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".nc")]
        )
        self.input_size = input_size
        self.target_size = target_size
        self.use_condition = use_condition
        self.wavelet = wavelet
        self.gamma = gamma

        # PPTSRDataset 类似: gamma=(x/255)^0.15，但这里假设你 clip 到[0,999] => (x/999)^0.15
        self.gamma_div = 999.0
        self.gamma_exp = 0.15

        # 若 use_condition=True，下面这些变量会被读取并拼接到 lr
        self.variables = ["r2", "t", "u10", "v10", "lsm", "z"]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        ds = xr.open_dataset(file_path)

        # ---------------- (1) 读取高分辨率降水 (hr) ----------------
        if "acpcp" not in ds:
            ds.close()
            raise ValueError(f"{file_path} 缺少 'acpcp' 变量!")
        ppt_arr = ds["acpcp"].values  # shape: (H, W) typically (224,224)
        ds.close()

        # Clip 降水到 [0, 999]，以防极端值（跟你原先做法一致）
        ppt_arr = np.clip(ppt_arr, 0, 999)

        # 转成张量 => (1,H,W)
        hr = torch.tensor(ppt_arr, dtype=torch.float32).unsqueeze(0)

        # ---------------- (2) 构造低分辨率 lr ----------------
        #   先下采样到 (input_size), 再上采样回 (target_size)
        #   这里下采样用 bicubic, 上采样用 nearest(与论文类似)
        hr_unsq = hr.unsqueeze(0)  # (1,1,H,W)
        downsampled = F.interpolate(
            hr_unsq, size=self.input_size, mode="bicubic", align_corners=False
        )
        lr_up = F.interpolate(
            downsampled, size=self.target_size, mode="nearest", align_corners=None
        )
        lr = lr_up.squeeze(0)  # => shape (1,H,W)

        # ---------------- (3) 若 use_condition=True，则读取其它变量并拼接到 lr ----------------
        if self.use_condition:
            ds_cond = xr.open_dataset(file_path)
            cond_list = []
            for var in self.variables:
                if var not in ds_cond:
                    ds_cond.close()
                    raise ValueError(f"{file_path} 缺少变量 {var}")

                var_arr = ds_cond[var].values  # shape (H,W)
                var_arr = np.nan_to_num(var_arr, nan=0.0)

                # 论文中 topo 的处理: topo = (topo - min)/(max-min)
                # 这里一样做 min-max scale, 你也可以换成其他方式
                vmin = var_arr.min()
                vmax = var_arr.max()
                rng = (vmax - vmin) + 1e-6
                var_arr = (var_arr - vmin) / rng

                # 重塑 var_arr 为 (H, W)，保留最后两维
                var_arr = var_arr.reshape(var_arr.shape[-2], var_arr.shape[-1])

                # 转为张量 => shape(1,H,W)
                var_tensor = torch.tensor(var_arr, dtype=torch.float32).unsqueeze(0)  # (1,H,W)
                cond_list.append(var_tensor)

            ds_cond.close()
            # 把所有条件变量在通道维拼起来 => shape(Ccond, H,W)
            cond_block = torch.cat(cond_list, dim=0)
            # 与lr(降水)在通道维拼 => shape(1+Ccond, H,W)
            lr = torch.cat([lr, cond_block], dim=0)

        # ---------------- (4) gamma 校正 (原论文对整张 hr & lr 都做) ----------------
        #   PPTSRDataset 是先 /255 再 ^0.15，这里我们把 /255 改成 /999
        #   注意：现在 lr 已经可能是多通道(降水+其他)，原论文只对降水做 gamma。
        #   若你只想对降水通道做，就仅处理 lr[0], hr[0]. 这里演示对所有通道都做。
        if self.gamma:
            hr = hr.clamp(0.0, self.gamma_div) / self.gamma_div
            hr = hr ** self.gamma_exp

            lr[0] = lr[0].clamp(0.0, self.gamma_div) / self.gamma_div
            lr[0] = lr[0] ** self.gamma_exp

        # ---------------- (5) 小波变换 wavelet ----------------
        #   论文做法: 对 hr、lr 分别执行 pywt.dwt2(array, 'haar') => 返回(LL, (LH, HL, HH))
        #   再用 np.concatenate(..., axis=0) 拼到一起 => 变成 4×(H/2)×(W/2).
        #
        #   如果 lr 有多通道(比如 7 通道), 需要对每个通道做 DWT2，然后把结果合并到通道维。
        #   PPTSRDataset 只示范了单通道 + topo(1通道) => 直接 dwt2(2D array).
        #   下面代码演示多通道的做法：对每个 channel 独立做 wavelet，并把 4 个子带拼回 channel。

        def dwt2_per_channel(tensor_3d):
            """
            对 (C,H,W) 的张量逐通道做 dwt2，然后把结果拼在 channel 维度。
            返回 shape => (4*C, H/2, W/2).
            """
            c, h, w = tensor_3d.shape
            wavelet_channels = []
            for i in range(c):
                arr_2d = tensor_3d[i].numpy()
                LL, (LH, HL, HH) = pywt.dwt2(arr_2d, 'haar')
                # 把四个子带在 0 维拼成(4, H/2, W/2)
                sub_bands = np.stack([LL, LH, HL, HH], axis=0)
                wavelet_channels.append(sub_bands)
            # wavelet_channels 是长度 = c，每个元素 shape = (4, H/2, W/2)
            wavelet_channels = np.concatenate(wavelet_channels, axis=0)  # => (4*c, H/2, W/2)
            return torch.tensor(wavelet_channels, dtype=torch.float32)

        if self.wavelet:
            # hr shape: (1,H,W) or (C,H,W) => 但这里理应只有1通道(降水)
            hr = dwt2_per_channel(hr)

            # lr 可能有多个通道 (降水 + condition) => 对每个通道做 DWT2
            lr = dwt2_per_channel(lr)

        # ---------------- (6) 返回与论文PPTSRDataset类似的字典 ----------------
        return {"hr": hr, "lr": lr}


class PrecipitationDataModule(LightningDataModule):
    def __init__(self, train_dir, val_dir, test_dir, batch_size, large_size, small_size, use_condition,
                 wavelet=True, gamma=True, num_workers=4):
        """
        Args:
            train_dir, val_dir, test_dir: 训练/验证/测试数据集目录
            batch_size: batch 大小
        """
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.large_size = large_size
        self.small_size = small_size
        self.use_condition = use_condition
        self.wavelet = wavelet
        self.gamma = gamma
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = PrecipitationDataset(
                data_dir=self.train_dir,
                input_size=self.small_size,
                target_size=self.large_size,
                use_condition=self.use_condition,   # 是否使用多变量作为条件
                wavelet=self.wavelet,         # 是否对每个通道做小波
                gamma=self.gamma            # 是否做 gamma 校正
            )
            self.val_dataset = PrecipitationDataset(
                data_dir=self.val_dir,
                input_size=self.small_size,
                target_size=self.large_size,
                use_condition=self.use_condition,
                wavelet=self.wavelet,
                gamma=self.gamma
            )

        if stage == "test":
            self.test_dataset = PrecipitationDataset(
                data_dir=self.test_dir,
                input_size=(28, 28),
                target_size=(224, 224),
                use_condition=self.use_condition,
                wavelet=self.wavelet,
                gamma=self.gamma
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )