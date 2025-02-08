import torch
from pytorch_lightning import LightningModule
from torch.optim import Adam
from skimage.metrics import structural_similarity as compare_ssim, peak_signal_noise_ratio as calculate_psnr
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.cvae import cVAE
from src.cvae_loss import lossVAE


class cVAETrainer(LightningModule):
    def __init__(self, spatial_x_dim, out_dim, learning_rate=0.00001):
        super().__init__()
        self.model = cVAE(spatial_x_dim, out_dim)
        self.learning_rate = learning_rate

    def forward(self, x, y):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_prime, mu, logvar = self(x, y)
        loss = lossVAE(y, y_prime, mu, logvar, self.current_epoch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_prime, mu, logvar = self(x, y)
        loss = lossVAE(y, y_prime, mu, logvar, self.current_epoch)
        y_prime_np = y_prime.cpu().numpy().squeeze(0)
        y_np = y.cpu().numpy().squeeze(0)
        psnr = calculate_psnr(y_prime_np, y_np, data_range=y_np.max() - y_np.min())
        ssim = compare_ssim(y_np, y_prime_np, data_range=y_np.max() - y_np.min())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_psnr', round(psnr, 4), prog_bar=True)
        self.log('val_ssim', round(ssim, 4), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_prime, mu, logvar = self(x, y)
        y_prime_np = y_prime.cpu().numpy().squeeze()
        y_np = y.cpu().numpy().squeeze()
        psnr = calculate_psnr(y_prime_np, y_np, data_range=1)
        ssim = compare_ssim(y_np, y_prime_np, data_range=1)
        self.log('test_psnr', psnr)
        self.log('test_ssim', ssim)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                      factor=0.5, patience=30,
                                      verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
