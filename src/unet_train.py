import torch
import torch.nn.functional as F
from torch.optim import AdamW
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

# 导入你自己实现的数据模块与模型（请根据实际工程目录调整导入路径）
from src.dataset.ddpm_dataset import RobustPrecipitationDataModule
from src.models import Network


# =============================================================================
# 定义 LightningModule：封装 UNet 的训练、验证及采样逻辑
# =============================================================================
class UnetLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=3e-5):
        """
        构造 UNet 训练模块：
          - 使用 Network.UNet 作为模型，
          - 模型输入为 7 通道条件数据，输出为 1 通道预测的残差；
          - 目标为标准化后的残差（fine - coarse）。
        """
        super().__init__()
        self.learning_rate = learning_rate

        # 初始化 UNet 模型。这里设置：
        #   - img_resolution 为 (224, 224)
        #   - in_channels = 7（即数据集中的条件通道）
        #   - out_channels = 1（预测残差）
        #   - label_dim = 0（本实验无额外条件标签）
        #   - use_diffuse = False（非扩散模型训练）
        self.model = Network.UNet(
            img_resolution=(224, 224),
            in_channels=7,
            out_channels=1,
            label_dim=0,
            use_diffuse=False
        )

        self.loss_fn = torch.nn.MSELoss()
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse   = torchmetrics.MeanSquaredError()

        # 记录训练时每个 step 的 loss，用于 epoch 聚合
        self.train_loss_accum = []

    def forward(self, x, class_labels=None):
        # 这里直接调用 UNet 前向（本实验无额外条件标签，所以 class_labels 为 None）
        return self.model(x, class_labels=class_labels)

    def training_step(self, batch, batch_idx):
        # 从 batch 中取出条件数据与目标残差
        image_input = batch["inputs"]       # shape: (B, 7, 224, 224)
        image_target = batch["targets"]       # shape: (B, 1, 224, 224)

        output = self.forward(image_input)    # UNet 输出预测残差
        loss = self.loss_fn(output, image_target)
        loss = loss.mean()

        self.train_loss_accum.append(loss.detach())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_mse.update(output, image_target)
        return loss

    def on_train_epoch_end(self):
        if self.train_loss_accum:
            epoch_loss = sum(self.train_loss_accum) / len(self.train_loss_accum)
            self.log("train_epoch_loss", epoch_loss, prog_bar=True)
        self.train_loss_accum = []

    def validation_step(self, batch, batch_idx):
        image_input = batch["inputs"]
        image_target = batch["targets"]

        output = self.forward(image_input)
        loss = self.loss_fn(output, image_target)
        loss = loss.mean()

        # 动态计算当前 batch 的归一化后 target 的实际数值范围
        data_min = image_target.min()
        data_max = image_target.max()
        dynamic_range = (data_max - data_min).item()

        mse_val  = F.mse_loss(output, image_target)
        psnr_val = peak_signal_noise_ratio(output, image_target, data_range=dynamic_range)
        ssim_val = structural_similarity_index_measure(output, image_target, data_range=dynamic_range)

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_mse", mse_val, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_psnr", psnr_val, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_ssim", ssim_val, prog_bar=True, on_step=True, on_epoch=True)

        # 累积每个 validation step 的输出，用于 epoch 结束后计算聚合指标
        if not hasattr(self, "_val_outputs"):
            self._val_outputs = []
        self._val_outputs.append({"pred": output.detach(), "target": image_target.detach()})
        return loss

    def on_validation_epoch_end(self):
        if hasattr(self, "_val_outputs") and len(self._val_outputs) > 0:
            all_preds = torch.cat([x["pred"] for x in self._val_outputs], dim=0)
            all_targets = torch.cat([x["target"] for x in self._val_outputs], dim=0)
            overall_mse = F.mse_loss(all_preds, all_targets)
            self.log("epoch_val_mse", overall_mse, prog_bar=True)
            self._val_outputs = []

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    @torch.no_grad()
    def sample_model(self, dataloader):
        """
        模拟采样过程，用于生成预测图像进行可视化比较。
        假设数据集中的 sample 包含：
          - "inputs"：条件数据（7通道），
          - "coarse_acpcp"：经过标准化后的低分辨率降水数据，
          - "fine_acpcp"：标准化后的高分辨率真实降水数据。
        模型输出预测的残差，预测的高分辨率图像可通过 coarse_acpcp + residual 得到。
        """
        self.model.eval()
        batch = next(iter(dataloader))
        images_input = batch["inputs"].to(self.device)
        coarse = batch["coarse_acpcp"].to(self.device)
        fine = batch["fine_acpcp"].to(self.device)

        predicted_residual = self.forward(images_input)
        predicted_fine = coarse + predicted_residual

        # 这里假设数据集提供了绘图方法，你可以调用例如 dataset.plot_batch(coarse, fine, predicted_fine)
        # 作为示例，我们直接计算误差指标
        base_error = torch.mean(torch.abs(fine - coarse))
        pred_error = torch.mean(torch.abs(fine - predicted_fine))
        # 此处返回的 fig、ax 仅作占位，你可以根据实际需要构建绘图
        fig, ax = None, None
        return (fig, ax), (base_error.item(), pred_error.item())


# =============================================================================
# 主函数：使用 Lightning Trainer 进行训练，并添加 EarlyStopping 回调（patience=20）
# =============================================================================
if __name__ == "__main__":
    batch_size = 8
    num_epochs = 1000

    # 数据目录（请根据实际路径修改）
    train_dir = "../processed_datasets/new_train"
    val_dir   = "../processed_datasets/new_val"
    test_dir  = "../processed_datasets/new_test"

    dm = RobustPrecipitationDataModule(train_dir, val_dir, test_dir, batch_size=batch_size)
    model = UnetLightningModule(learning_rate=3e-5)

    # EarlyStopping 回调：监控 "val_loss"，patience 设为 20
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        precision="16-mixed",       # 混合精度训练
        accumulate_grad_batches=8,
        log_every_n_steps=10,
        default_root_dir="../logs/unet/"
    )
    trainer.fit(model, dm)

    # 保存最佳模型
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        model = UnetLightningModule.load_from_checkpoint(best_model_path)
        torch.save(model.state_dict(), "../models/unet/best_model.pt")