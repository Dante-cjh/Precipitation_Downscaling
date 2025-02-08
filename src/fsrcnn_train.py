import torch
from pytorch_lightning import LightningModule
from torch.nn import MSELoss
from torch.optim import Adam
from skimage.metrics import structural_similarity as compare_ssim, peak_signal_noise_ratio as calculate_psnr
from src.models.fsrcnn import FSRCNN_ESM


class FSRCNNTrainer(LightningModule):
    def __init__(self, scale_factor, learning_rate):
        super().__init__()
        self.model = FSRCNN_ESM(scale_factor)
        self.criterion = MSELoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self.model(lr)
        loss = self.criterion(sr, hr)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self.model(lr)
        loss = self.criterion(sr, hr)
        sr_np = sr.cpu().numpy().squeeze()
        hr_np = hr.cpu().numpy().squeeze()
        psnr = calculate_psnr(sr_np, hr_np, data_range=hr_np.max() - hr_np.min())
        ssim = compare_ssim(sr_np, hr_np, data_range=hr_np.max() - hr_np.min())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_psnr", psnr, prog_bar=True)
        self.log("val_ssim", ssim, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer