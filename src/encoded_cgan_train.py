import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from models.encoded_cgan import DiscModel, EncodedGenerator
from util import cosine_decay
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr


def train(train_dataset, val_dataset, batch_size, learning_rate, max_epochs, warmup_epochs, model_size, upscale_factor, cvars,
          log_transform, spatial_attention, channel_attention, weight_mse, output_dir, log_dir, results_dir, patience=10):
    # 设置设备
    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # 初始化模型
    generator = EncodedGenerator(in_ch=1, ncvar=len(cvars), use_ele=True, cam=channel_attention, sam=spatial_attention,
                                 stage_chs=[model_size // 2 ** d for d in range(4)])
    discriminator = DiscModel()

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # 多 GPU 支持
    if torch.cuda.device_count() > 1:
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)

    # 定义损失函数和优化器
    gen_criterion = torch.nn.L1Loss().to(device)
    disc_criterion = torch.nn.BCELoss().to(device)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    # 学习率调度器
    lr_decay_lambda = lambda epoch: cosine_decay(epoch, warmup=warmup_epochs, max_epoch=max_epochs)
    gen_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, lr_lambda=lr_decay_lambda)
    disc_scheduler = torch.optim.lr_scheduler.LambdaLR(disc_optimizer, lr_lambda=lr_decay_lambda)

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化 Tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # 开始训练
    for epoch in range(max_epochs + 1):
        generator.train()
        start_time = time.time()

        for lr_input, cvars, hr_target, hr_ele in tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}"):

            lr_input = lr_input.to(device)
            cvars = [cv.to(device) for cv in cvars]
            hr_target = hr_target.to(device)
            hr_ele = hr_ele.to(device)

            # Generator optimization
            gen_optimizer.zero_grad()
            pred = generator(lr_input, cvars, elevation=hr_ele)

            with torch.no_grad():
                adv_loss = -torch.mean(torch.log(discriminator(pred)))
            content_loss = gen_criterion(pred, hr_target)
            gen_loss = weight_mse * content_loss + adv_loss
            gen_loss.backward()
            gen_optimizer.step()
            gen_scheduler.step()

            # Discriminator optimization
            disc_optimizer.zero_grad()
            disc_input = torch.cat((hr_target, pred.detach()), dim=0)

            disc_pred = discriminator(disc_input)

            # 动态匹配判别器输出生成标签
            dsc_out_size = list(disc_pred.shape)
            dsc_out_size[0] = dsc_out_size[0] // 2
            dsc_out_size = tuple(dsc_out_size)
            true_label = torch.ones(dsc_out_size).to(device)
            false_label = torch.zeros(dsc_out_size).to(device)
            disc_label = torch.cat((true_label, false_label), dim=0)

            disc_loss = disc_criterion(disc_pred, disc_label) / 2
            disc_loss.backward()
            disc_optimizer.step()
            disc_scheduler.step()

        # 日志记录
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch}, Gen Loss: {gen_loss.item():.4f}, Disc Loss: {disc_loss.item():.4f}, Time: {elapsed_time:.2f}s")

        # Tensorboard 记录
        writer.add_scalar("Loss/Generator", gen_loss.item(), epoch)
        writer.add_scalar("Loss/Discriminator", disc_loss.item(), epoch)

        val_loss = validate(generator, val_dataset, device, epoch, log_transform, results_dir, writer)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(generator.state_dict(), f"{output_dir}/best_generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"{output_dir}/best_discriminator_epoch_{epoch}.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    writer.close()

def validate(generator, val_dataset, device, epoch, log_transform, results_dir, writer):
    generator.eval()
    best_psnr = -float('inf')
    best_pred_np = None
    best_hr_np = None

    # 如果results_dir不存在，则创建
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with torch.no_grad():
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
        total_psnr, total_ssim, total_mse, total_cc, count = 0, 0, 0, 0, 0

        for lr_input, cvars, hr_target, hr_ele in tqdm(val_dataloader, desc="Validation"):
            lr_input = lr_input.to(device)
            cvars = [cv.to(device) for cv in cvars]
            hr_target = hr_target.to(device)
            hr_ele = hr_ele.to(device)

            pred = generator(lr_input, cvars, elevation=hr_ele)

            pred_np = pred.cpu().numpy()
            hr_np = hr_target.cpu().numpy()

            if log_transform:
                pred_np = np.exp(pred_np) - 1
                hr_np = np.exp(hr_np) - 1

            mse = np.mean((hr_np - pred_np) ** 2)
            cc = np.mean([pearsonr(pred_np[i].flatten(), hr_np[i].flatten())[0] for i in range(pred_np.shape[0])])

            # 计算 PSNR 和 SSIM
            psnr_value = calculate_psnr(pred_np, hr_np, data_range=hr_np.max() - hr_np.min())
            ssim_value = compare_ssim(hr_np[0, 0], pred_np[0, 0], data_range=hr_np.max() - hr_np.min())

            total_psnr += psnr_value
            total_ssim += ssim_value
            total_mse += mse
            total_cc += cc
            count += 1

            # 保存最佳结果
            if psnr_value > best_psnr:
                best_psnr = psnr_value
                best_pred_np = pred_np
                best_hr_np = hr_np

        # 平均 PSNR 和 SSIM
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_mse = total_mse / count
        avg_cc = total_cc / count

        # 记录到 Tensorboard
        writer.add_scalar("Metrics/PSNR", avg_psnr, epoch)
        writer.add_scalar("Metrics/SSIM", avg_ssim, epoch)
        writer.add_scalar("Metrics/MSE", avg_mse, epoch)
        writer.add_scalar("Metrics/CC", avg_cc, epoch)

        # 保存最佳结果
        if best_pred_np is not None and best_hr_np is not None:
            save_image(best_hr_np[0, 0], f"{results_dir}/best_hr_epoch_{epoch}.png")
            save_image(best_pred_np[0, 0], f"{results_dir}/best_sr_epoch_{epoch}.png")
            writer.add_image("Best/HR", best_hr_np[0, 0], epoch, dataformats='HW')
            writer.add_image("Best/SR", best_pred_np[0, 0], epoch, dataformats='HW')


    return total_mse / count

def save_image(array, path):
    import matplotlib.pyplot as plt
    plt.imshow(array, cmap='viridis')
    plt.colorbar()
    plt.savefig(path)
    plt.close()
