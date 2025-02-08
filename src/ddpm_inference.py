import torch
import numpy as np
import matplotlib.pyplot as plt
# 注意：请根据你项目的实际目录结构调整下面两个导入路径
from src.models import Network
from src.dataset.ddpm_dataset import RobustPrecipitationZScoreDataset


# -----------------------------------------------------------------------------
# 使用 UNet 模型进行采样（非扩散模型）
@torch.no_grad()
def sample_unet(input_batch, model, device, dataset):
    """
    使用 UNet 模型采样：
      - 从 input_batch 中获取条件数据和用于展示的 low/high-res 降水图像；
      - 调用模型生成预测残差，并利用 dataset.residual_to_fine_image 恢复出 fine 图像。
    """
    # 获取条件数据与可视化数据
    images_input = input_batch["inputs"].to(device)  # 7通道条件数据
    coarse = input_batch["coarse_acpcp"].to(device)  # 1通道 low-res 降水图
    fine = input_batch["fine_acpcp"].to(device)  # 1通道真实 fine 降水图

    # UNet 模型前向，注意本模型不使用额外条件标签，因此传入 None
    residual = model(images_input, class_labels=None)
    # 根据数据集中定义的接口，将预测残差加到 coarse 图像上恢复出 fine 图像
    predicted = dataset.residual_to_fine_image(residual.detach().cpu(), coarse.cpu())
    return coarse.cpu(), fine.cpu(), predicted


# -----------------------------------------------------------------------------
# 使用扩散模型进行采样（按照论文中 EDMPrecond 采样流程）
@torch.no_grad()
def sample_diffusion(input_batch, model, device, dataset, num_steps=40,
                     sigma_min=0.002, sigma_max=80, rho=7, S_churn=40,
                     S_min=0, S_max=float('inf'), S_noise=1):
    """
    使用扩散模型采样：
      - 从 input_batch 中获取条件数据和可视化数据；
      - 采用连续噪声尺度 sigma 生成初始噪声，逐步采样去噪生成预测残差；
      - 利用 dataset.residual_to_fine_image 将预测残差与 coarse 图像相加恢复 fine 图像。
    """
    images_input = input_batch["inputs"].to(device)  # 7通道条件数据
    coarse = input_batch["coarse_acpcp"].to(device)  # 1通道 low-res 降水图
    fine = input_batch["fine_acpcp"].to(device)  # 1通道真实 fine 降水图

    # 如果模型定义了 sigma_min/sigma_max，则使用它们
    if hasattr(model, "sigma_min"):
        sigma_min = max(sigma_min, model.sigma_min)
    if hasattr(model, "sigma_max"):
        sigma_max = min(sigma_max, model.sigma_max)

    # 初始噪声：形状与输出残差相同（B, 1, H, W）
    init_noise = torch.randn((images_input.shape[0], 1, images_input.shape[2],
                              images_input.shape[3]),
                             dtype=torch.float64, device=device)

    # 离散化噪声尺度（连续 sigma schedule）
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=init_noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    # 如果模型提供了 round_sigma 方法，则使用
    if hasattr(model, "round_sigma"):
        t_steps = torch.cat([model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    else:
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    # 主采样循环：从初始噪声逐步去噪
    x_next = init_noise * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if (S_min <= t_cur <= S_max) else 0
        if hasattr(model, "round_sigma"):
            t_hat = model.round_sigma(t_cur + gamma * t_cur)
        else:
            t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + ((t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur))
        denoised = model(x_hat, t_hat, images_input, None).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            denoised = model(x_next, t_next, images_input, None).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    # 将最终预测残差与 coarse 图像相加恢复 fine 图像
    predicted = dataset.residual_to_fine_image(x_next.detach().cpu(), coarse.cpu())
    return coarse.cpu(), fine.cpu(), predicted


# -----------------------------------------------------------------------------
# 主程序入口
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----------------------
    # 加载扩散模型
    diff_model = Network.EDMPrecond(
        img_resolution=(224, 224),
        in_channels=8,  # 扩散模型训练时使用 7 条件通道 + 1 残差通道输入
        out_channels=1,
        label_dim=0,
        use_diffuse=True
    ).to(device)
    diff_model.load_state_dict(torch.load("../models/unet_diffusion/best_model.pt", map_location=device))

    # ----------------------
    # 加载 UNet 模型
    unet_model = Network.UNet(
        img_resolution=(224, 224),
        in_channels=7,  # UNet 模型直接以 7 通道条件数据为输入
        out_channels=1,
        label_dim=0,
        use_diffuse=False
    ).to(device)
    unet_model.load_state_dict(torch.load("../models/unet/best_model.pt", map_location=device))

    # ----------------------
    # 加载测试数据集：请确保测试目录中存放的是你处理好的 .nc 文件
    test_dir = "../processed_datasets/new_test"
    dataset_test = RobustPrecipitationZScoreDataset(test_dir)

    # 构造一个 mini-batch（例如取前 4 个样本）
    batch_samples = [dataset_test[i] for i in range(4)]
    batch = {}
    for key in batch_samples[0].keys():
        batch[key] = torch.stack([sample[key] for sample in batch_samples])

    # ----------------------
    # 尝试扩散模型采样
    coarse_diff, fine_diff, predicted_diff = sample_diffusion(batch, diff_model, device, dataset_test)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(coarse_diff[0, 0].cpu(), cmap="viridis")
    ax[0].set_title("Coarse")
    ax[1].imshow(fine_diff[0, 0].cpu(), cmap="viridis")
    ax[1].set_title("Fine")
    ax[2].imshow(predicted_diff[0, 0].cpu(), cmap="viridis")
    ax[2].set_title("Predicted (Diffusion)")
    plt.show()

    # ----------------------
    # 尝试 UNet 模型采样
    coarse_unet, fine_unet, predicted_unet = sample_unet(batch, unet_model, device, dataset_test)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(coarse_unet[0, 0].cpu(), cmap="viridis")
    ax[0].set_title("Coarse")
    ax[1].imshow(fine_unet[0, 0].cpu(), cmap="viridis")
    ax[1].set_title("Fine")
    ax[2].imshow(predicted_unet[0, 0].cpu(), cmap="viridis")
    ax[2].set_title("Predicted (UNet)")
    plt.show()