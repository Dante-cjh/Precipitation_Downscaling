import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.dataset.ddpm_dataset import RobustPrecipitationDataModule
from src.models import NetworkV3

class SaveEveryNEpochs(pl.callbacks.Callback):
    def __init__(self, save_dir, save_every_n_epochs=20):
        self.save_dir = save_dir
        self.save_every_n_epochs = save_every_n_epochs
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.save_every_n_epochs == 0:
            save_path = os.path.join(self.save_dir, f"model_epoch_{epoch + 1}.pt")
            # 保存内部模型的 state_dict 而非整个 LightningModule 的 state_dict
            torch.save(pl_module.model.state_dict(), save_path)


# =============================================================================
# 1. 定义损失函数 EDMLoss（与原始实现一致）
# =============================================================================
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, conditional_img=None, labels=None, augment_pipe=None):
        # 生成噪声尺度 sigma
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        if augment_pipe is not None:
            y, augment_labels = augment_pipe(images)
        else:
            y, augment_labels = images, None
        # 加入噪声后前向计算
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, conditional_img, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


# =============================================================================
# 2. 定义 LightningModule，封装模型、训练、验证逻辑
# =============================================================================
class DiffusionLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        """
        这里构造的模型使用 Network.EDMPrecond（论文中的扩散模型实现），
        模型输入为 7 通道条件数据（低分辨率降水及其它条件数据），输出为 1 通道（降水残差）。
        """
        super().__init__()
        self.learning_rate = learning_rate

        # 初始化模型（模型代码在 Network 模块中）
        # 注意：img_resolution 为 (224, 224)，in_channels=8 （7个条件通道+1个目标残差通道作为条件输入）
        self.model = NetworkV3.EDMPrecond(
            img_resolution=(224, 224),
            in_channels=8,  # 7 通道条件数据 + 1 通道降水残差（作为条件输入）
            out_channels=1,
            label_dim=0,
            use_diffuse=True
        )

        # 损失函数
        self.loss_fn = EDMLoss()

        # 定义指标（这里只对训练集使用 MSE 指标累积）
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse   = torchmetrics.MeanSquaredError()
        self.train_loss_accum = []  # 用于累积每个 step 的 train loss

    def forward(self, x, sigma, conditional_img, labels=None):
        return self.model(x, sigma, conditional_img, labels)

    def training_step(self, batch, batch_idx):
        # batch 中包含 "inputs" (7 通道条件数据) 和 "targets" (1 通道标准化后的残差)
        image_input  = batch["inputs"]
        image_target = batch["targets"]

        loss = self.loss_fn(
            net=self.model,
            images=image_target,
            conditional_img=image_input,
            labels=None
        )
        loss = loss.mean()
        self.train_loss_accum.append(loss.detach())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self.train_loss_accum:
            epoch_loss = sum(self.train_loss_accum) / len(self.train_loss_accum)
            self.log("train_epoch_loss", epoch_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.train_loss_accum = []

    def validation_step(self, batch, batch_idx):
        image_input  = batch["inputs"]
        image_target = batch["targets"]

        loss = self.loss_fn(
            net=self.model,
            images=image_target,
            conditional_img=image_input,
            labels=None
        )
        loss = loss.mean()

        # 获取预测结果
        pred = self._get_pred(image_input, image_target)

        # 动态计算归一化后图像的实际范围
        data_min = image_target.min()
        data_max = image_target.max()
        dynamic_range = (data_max - data_min).item()  # 例如如果大部分数据在 [-3, 3]，则 dynamic_range≈6

        mse_val  = F.mse_loss(pred, image_target)
        psnr_val = peak_signal_noise_ratio(pred, image_target, data_range=dynamic_range)
        ssim_val = structural_similarity_index_measure(pred, image_target, data_range=dynamic_range)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mse", mse_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_psnr", psnr_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_ssim", ssim_val, prog_bar=True, on_step=False, on_epoch=True)

        if not hasattr(self, '_val_outputs'):
            self._val_outputs = []
        self._val_outputs.append({'pred': pred.detach(), 'target': image_target.detach()})

        return loss

    def on_validation_epoch_end(self):
        if hasattr(self, '_val_outputs') and len(self._val_outputs) > 0:
            all_preds = torch.cat([x['pred'] for x in self._val_outputs], dim=0)
            all_targets = torch.cat([x['target'] for x in self._val_outputs], dim=0)
            overall_mse = F.mse_loss(all_preds, all_targets)
            self.log("epoch_val_mse", overall_mse, prog_bar=True)
            self._val_outputs = []

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        # Add ReduceLROnPlateau scheduler
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True),
            'monitor': 'val_loss',
            'frequency': 1
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def _get_pred(self, image_input, image_target):
        """
        模拟采样过程：生成噪声尺度 sigma，再对 (target+n) 调用模型得到预测结果。
        """
        rnd_normal = torch.randn([image_target.shape[0], 1, 1, 1], device=image_target.device)
        sigma = (rnd_normal * self.loss_fn.P_std + self.loss_fn.P_mean).exp()
        y = image_target
        n = torch.randn_like(y) * sigma
        D_yn = self.model(y + n, sigma, image_input, None)
        return D_yn


# =============================================================================
# 3. 主函数，使用 Lightning Trainer 进行训练，并添加 EarlyStopping 回调（patience=10）
# =============================================================================
if __name__ == "__main__":
    # 数据目录（请根据实际路径修改）
    train_dir = "../processed_datasets/new_train"
    val_dir   = "../processed_datasets/new_val"
    test_dir  = "../processed_datasets/new_test"

    # 如果你预先计算了归一化参数，可以将它们传入数据集构造函数
    norm_means = {'acpcp': 10.540719985961914, 'lsm': 0.31139194936462644, 'r2': 76.87171173095703,
                  't': 281.4097900390625, 'u10': 0.6429771780967712, 'v10': -0.13687361776828766,
                  'z': 2719.8411298253704}
    norm_stds = {'acpcp': 24.950708389282227, 'lsm': 0.44974926605736953, 'r2': 17.958908081054688,
                 't': 18.581451416015625, 'u10': 5.890835762023926, 'v10': 4.902953624725342, 'z': 6865.68850114493}
    residual_mean = 0.0020228675566613674
    residual_std = 6.931405067443848

    dm = RobustPrecipitationDataModule(train_dir, val_dir, test_dir, batch_size=8,
                                       norm_means=norm_means, norm_stds=norm_stds, residual_mean=residual_mean, residual_std=residual_std)
    model = DiffusionLightningModule(learning_rate=1e-4)

    # EarlyStopping 回调（监控 "val_loss"，patience 设为 10）
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")
    save_every_n_epochs_callback = SaveEveryNEpochs(save_dir="../models/unet_diffusion_v3/", save_every_n_epochs=20)

    trainer = pl.Trainer(
        max_epochs=1000,
        callbacks=[early_stop_callback, checkpoint_callback, save_every_n_epochs_callback],
        precision="16-mixed",                      # 使用 16-mixed 精度
        accumulate_grad_batches=8,
        log_every_n_steps=10,
        default_root_dir="../logs/unet_diffusion_v3/"
    )
    trainer.fit(model, dm)

    # 保存最佳模型
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        best_model = DiffusionLightningModule.load_from_checkpoint(best_model_path)
        torch.save(best_model.model.state_dict(), "../models/unet_diffusion_v3/best_model.pt")