import copy

import blobfile as bf
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

INITIAL_LOG_LOSS_SCALE = 20.0

class DDPM(pl.LightningModule):
    def __init__(
        self,
        model,
        diffusion,
        lr,
        ema_rate,
        schedule_sampler=None,
        weight_decay=0.0,
        model_dir="checkpoints",
    ):
        super(DDPM, self).__init__()
        self.model = model
        self.diffusion = diffusion
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay

        self.model_params = list(self.model.parameters())
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.ema_params = [
            copy.deepcopy(self.model_params) for _ in range(len(self.ema_rate))
        ]
        self.model_dir = model_dir

        # 关闭 Lightning 默认的自动优化, 我们会手动 backward & step
        self.automatic_optimization = False

        # ============ 累计器，用于按 epoch 计算平均值 ============
        self.train_loss_sum = 0.0
        self.train_rmse_sum = 0.0
        self.train_steps = 0

        self.val_loss_sum = 0.0
        self.val_rmse_sum = 0.0
        self.val_psnr_sum = 0.0
        self.val_ssim_sum = 0.0
        self.val_steps = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.5, 0.9),
            weight_decay=self.weight_decay,
        )
        step_lr = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[150, 300], gamma=0.1
        )
        # cos_lr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=50, T_mult=2, eta_min=1e-6
        # )
        # 这里先返回 step_lr 作为调度器
        return {
            "optimizer": optimizer,
            "lr_scheduler": step_lr,
        }

    def log_loss_dict(self, diffusion, ts, losses, stage: str):
        # 原作者的 batch 级别日志；可以保留或精简
        for key, values in losses.items():
            self.log(f"{stage}_{key}", values.mean().item(), prog_bar=True, on_step=True, on_epoch=False)
            # 下面 log quartile 细分可以自行去掉，防止日志爆炸
            # for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            #     quartile = int(4 * sub_t / diffusion.num_timesteps)
            #     self.log(f"{stage}_{key}_q{quartile}", sub_loss, on_step=True, on_epoch=False)

    def shared_step(self, batch, batch_idx, stage: str):
        hr, lr = batch["hr"], batch["lr"]
        # 采样时间步
        t, weights = self.schedule_sampler.sample(hr.shape[0], device=hr.device)

        # 调用 diffusion.training_losses(model, hr, t, model_kwargs=...)
        # 它应该返回一个 dict，比如 { "loss": ..., "pred_xstart": ... } etc.
        losses = self.diffusion.training_losses(
            self.model, hr, t, model_kwargs={"low_res": lr}
        )
        # 如果是 LossAwareSampler，则更新
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_all_losses(t, losses["loss"].detach())

        # 总体 loss
        loss = (losses["loss"] * weights).mean()

        losses_for_log = {}
        for k, v_tensor in losses.items():
            # v_tensor: shape = (B, C, 112, 112) or (B,) ...
            # weights: shape = (B,) (希望如此)
            if weights.dim() == 1 and v_tensor.dim() == 4:
                # 让 weights 在 C/H/W 维上广播
                w = weights.view(-1, 1, 1, 1)
                losses_for_log[k] = v_tensor * w
            else:
                # 如果是 (B,) × (B,) 也行
                # 或者看看实际dim是啥
                losses_for_log[k] = v_tensor * weights

        self.log_loss_dict(
            self.diffusion, t, losses_for_log, stage
        )

        # ============ 计算 RMSE/PSNR/SSIM =============
        # 需 pred_xstart 代表网络预测出的 HR
        rmse, psnr, ssim = 0.0, 0.0, 0.0
        if "pred_xstart" in losses:
            pred = losses["pred_xstart"]  # shape 与 hr 相同
            # RMSE
            mse_val = F.mse_loss(pred, hr)
            rmse = torch.sqrt(mse_val)
            # PSNR (假设 hr/pred 在 [0,1] or 其他范围)
            # data_range = 1.0 如果确认是 [0,1]
            # 若不确定, 也可用  hr.max() - hr.min()
            data_range = 1.0
            psnr = peak_signal_noise_ratio(pred, hr, data_range=data_range)
            ssim = structural_similarity_index_measure(pred, hr, data_range=data_range)

        return loss, rmse, psnr, ssim

    def training_step(self, batch, batch_idx):
        # shared_step 返回 (loss, rmse, psnr, ssim)
        loss, rmse, psnr, ssim = self.shared_step(batch, batch_idx, "train")

        # 手动优化流程
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)

        # 记录梯度范数
        sqsum = 0.0
        for p in self.model_params:
            if p.grad is not None:
                sqsum += float((p.grad**2).sum().item())
        self.log("grad_norm", np.sqrt(sqsum), prog_bar=True, on_step=True, on_epoch=False)

        opt.step()

        # 如果是 epoch 最后一个 batch，就手动 step lr_scheduler
        if self.trainer.is_last_batch:
            sch = self.lr_schedulers()
            sch.step()
            self.log("lr", sch.get_last_lr()[0], prog_bar=True, on_step=False, on_epoch=True)

        # EMA 更新
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model_params, rate=rate)

        # =========== 累计 epoch 级别的 train loss / rmse ===========
        self.train_loss_sum += loss.detach().item()
        self.train_rmse_sum += rmse if isinstance(rmse, float) else rmse.detach().item()
        self.train_steps += 1

        # training_step 必须返回 loss (Lightning约定)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # 同理
        loss, rmse, psnr, ssim = self.shared_step(batch, batch_idx, "val")

        self.val_loss_sum += loss.detach().item()
        self.val_rmse_sum += rmse if isinstance(rmse, float) else rmse.detach().item()
        self.val_psnr_sum += psnr if isinstance(psnr, float) else psnr.detach().item()
        self.val_ssim_sum += ssim if isinstance(ssim, float) else ssim.detach().item()
        self.val_steps += 1

        return loss

    # ==================== 每个 Epoch 结束后，记录平均指标 ====================

    def on_train_epoch_end(self):
        if self.train_steps > 0:
            avg_loss = self.train_loss_sum / self.train_steps
            avg_rmse = self.train_rmse_sum / self.train_steps

            # log到 Lightning 的默认logs
            self.log("Train/Epoch_Loss", avg_loss, prog_bar=True, on_epoch=True)
            self.log("Train/Epoch_RMSE", avg_rmse, prog_bar=True, on_epoch=True)

        # 清空计数器
        self.train_loss_sum = 0.0
        self.train_rmse_sum = 0.0
        self.train_steps = 0

    def on_validation_epoch_end(self):
        if self.val_steps > 0:
            avg_loss = self.val_loss_sum / self.val_steps
            avg_rmse = self.val_rmse_sum / self.val_steps
            avg_psnr = self.val_psnr_sum / self.val_steps
            avg_ssim = self.val_ssim_sum / self.val_steps

            self.log("Val/Epoch_Loss", avg_loss, prog_bar=True, on_epoch=True)
            self.log("Val/Epoch_RMSE", avg_rmse, prog_bar=True, on_epoch=True)
            self.log("Val/Epoch_PSNR", avg_psnr, prog_bar=True, on_epoch=True)
            self.log("Val/Epoch_SSIM", avg_ssim, prog_bar=True, on_epoch=True)

        self.val_loss_sum = 0.0
        self.val_rmse_sum = 0.0
        self.val_psnr_sum = 0.0
        self.val_ssim_sum = 0.0
        self.val_steps = 0

    def on_save_checkpoint(self, checkpoint):
        self.save()

    def get_blob_logdir(self):
        return self.model_dir

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if not rate:
                filename = f"model{(self.current_epoch):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.current_epoch):06d}.pt"
            with bf.BlobFile(bf.join(self.get_blob_logdir(), filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(0, self.model_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

    def _master_params_to_state_dict(self, model_params):
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = model_params[i]
        return state_dict