import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# =========== 1. 引入自己的 Dataset ===========
#   (假设在 your_dataset_file.py 中有 PrecipitationDataset 类)
from dataset.bias_iddpm_dataset import PrecipitationDataset, PrecipitationDataModule

# =========== 2. 引入 improved_diffusion 模块 ===========
#   (假设你已将 improved_diffusion 文件夹放在可以 import 的路径下)
from improved_diffusion.ddpm import DDPM
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    sr_create_model_and_diffusion,
    sr_model_and_diffusion_defaults,
)


# =========== 3. 自定义回调，间隔 N epoch 保存模型 ===========
class SaveEveryNEpochs(pl.Callback):
    def __init__(self, save_dir, save_every_n_epochs=20):
        super().__init__()
        self.save_dir = save_dir
        self.save_every_n_epochs = save_every_n_epochs
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.save_every_n_epochs == 0:
            save_path = os.path.join(self.save_dir, f"model_epoch_{epoch + 1}.pt")
            # 这里的 pl_module 通常是 DDPM LightningModule
            # 如果你想直接保存 pl_module.model 里的权重，可以改成：
            torch.save(pl_module.model.state_dict(), save_path)
            print(f"[SaveEveryNEpochs] Model saved at epoch {epoch + 1} to {save_path}")


def main():
    # =========== 4. 直接在代码里固定数据路径和超参数 ===========
    train_dir = "../processed_datasets/new_train"
    val_dir   = "../processed_datasets/new_val"
    test_dir  = "../processed_datasets/new_test"
    batch_size = 16

    # large_size 和 small_size 决定 HR & LR 尺寸
    large_size = 224
    small_size = 28

    # （可根据需要调参）
    lr = 1e-4
    weight_decay = 0.0
    ema_rate = "0.9999"
    max_epochs = 1000

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # =========== 5. 创建模型 & Diffusion ===========
    #   假设你想要 1 通道降水 + 6 个条件通道（use_condition=True） => in_channels=1, cond_channels=6
    #   或者你自行修改 cond_channels
    in_channels = 1
    cond_channels = 7  # 如果 PrecipitationDataset 拼接了 6 个气象变量
    # 可以根据 sr_model_and_diffusion_defaults() 的键来设置/覆盖
    model_conf = sr_model_and_diffusion_defaults()
    # 覆盖你需要的关键参数
    model_conf["large_size"] = large_size
    model_conf["small_size"] = small_size
    model_conf["num_channels"] = 128
    model_conf["num_res_blocks"] = 2
    model_conf["diffusion_steps"] = 4000
    model_conf["noise_schedule"] = "linear"
    model_conf["learn_sigma"] = True
    model_conf["class_cond"] = False
    model_conf["wavelet"] = True  # 如果你需要

    print("[main] Creating SR model & diffusion ...")
    model, diffusion = sr_create_model_and_diffusion(
        in_channels=in_channels,
        cond_channels=cond_channels,
        **model_conf,
    )
    model.to(device)

    # =========== 6. 创建 schedule_sampler ===========
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    # =========== 7. 构造自己的 Dataset & DataLoader ===========
    data_module = PrecipitationDataModule(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=batch_size,
        large_size=large_size,
        small_size=small_size,
        use_condition=True,
        wavelet=True,
        gamma=True,
        num_workers=4
    )

    # =========== 8. 用 improved_diffusion.ddpm 中的 DDPM LightningModule  ===========
    #   DDPM 的 init:
    #     DDPM(model, diffusion, lr, ema_rate, schedule_sampler, weight_decay=0.0, model_dir='')
    #   你可查看 super_res_train_lightning.py 了解更多细节
    model_dir = f"../models/bias_diffusion/"
    os.makedirs(model_dir, exist_ok=True)

    print("[main] Creating DDPM LightningModule ...")
    ddpm_module = DDPM(
        model=model,
        diffusion=diffusion,
        lr=lr,
        ema_rate=ema_rate,
        schedule_sampler=schedule_sampler,
        weight_decay=weight_decay,
        model_dir=model_dir
    )

    # =========== 9. 创建回调：EarlyStopping, ModelCheckpoint, SaveEveryNEpochs ===========
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="Val/Epoch_Loss",  # DDPM 里默认 log 的 validation loss key
        patience=10,
        mode="min"
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="{epoch}-{Loss:.4f}",
        monitor="Val/Epoch_Loss",
        save_top_k=1,      # 只保留最优
        mode="min"
    )
    save_every_n_epochs_callback = SaveEveryNEpochs(
        save_dir=model_dir,
        save_every_n_epochs=20
    )

    callbacks = [early_stop_callback, checkpoint_callback, save_every_n_epochs_callback]


    # =========== 10. 创建 Trainer 并开始训练 ===========
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        log_every_n_steps=10,
        default_root_dir="../logs/bias_diffusion",  # 日志保存目录
    )

    print("[main] Starting Training ...")
    trainer.fit(ddpm_module, datamodule=data_module)

    # =========== 12. 训练结束后，可选地测试一下 ===========
    print("[main] Training complete. Now testing best model if needed ...")
    # 载入最优模型
    if checkpoint_callback.best_model_path:
        best_ckpt = checkpoint_callback.best_model_path
        best_model = DDPM.load_from_checkpoint(best_ckpt, model=model, diffusion=diffusion, lr=lr,
                                               ema_rate=ema_rate,
                                               schedule_sampler=None,
                                               weight_decay=0.0,
                                               model_dir="checkpoints")
        trainer.test(best_model, datamodule=data_module)
        # 手动再保存一份 best model 的权重
        torch.save(best_model.model.state_dict(), os.path.join(model_dir, "best_model.pt"))
        print(f"[main] Best model saved to {os.path.join(model_dir, 'best_model.pt')}")

if __name__ == "__main__":
    # 可选：设置矩阵乘法精度
    torch.set_float32_matmul_precision("high")
    main()