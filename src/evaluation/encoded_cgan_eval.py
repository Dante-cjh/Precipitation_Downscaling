import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from scipy.stats import pearsonr

# 导入你定义的生成器模型和数据集类
from src.models.encoded_cgan import EncodedGenerator
from src.dataset.encoded_cgan_dataset import PrecipitationDatasetCGAN

def save_image(array, path, cmap='viridis'):
    """
    使用 matplotlib 保存图像（二维数组）。
    """
    plt.imshow(array, cmap=cmap)
    plt.colorbar()
    plt.savefig(path)
    plt.close()

def calculate_metrics(pred, target, threshold=0.1):
    pred_binary = (pred > threshold).astype(int)
    target_binary = (target > threshold).astype(int)
    NA = ((pred_binary == 1) & (target_binary == 1)).sum()
    NB = ((pred_binary == 1) & (target_binary == 0)).sum()
    NC = ((pred_binary == 0) & (target_binary == 1)).sum()
    ND = ((pred_binary == 0) & (target_binary == 0)).sum()
    PC = (NA + ND) / (NA + NB + NC + ND) * 100 if (NA + NB + NC + ND) > 0 else 0
    PO = NC / (NA + NC) * 100 if (NA + NC) > 0 else 0
    FAR = NB / (NA + NB) * 100 if (NA + NB) > 0 else 0
    return PC, PO, FAR

def evaluate_encoded_cgan(test_data_dir, model_path, output_dir,
                          input_size=(28, 28), target_size=(224, 224),
                          upscale_factor=8, in_depth=1,
                          cvars=['r2', 't', 'u10', 'v10', 'lsm'],
                          log_transform=False,
                          spatial_attention=True, channel_attention=True,
                          model_size=256, batch_size=1, device=None):
    """
    对 encoded-cGAN 模型进行评估

    参数说明：
      test_data_dir: 测试数据所在目录，数据格式与训练时保持一致
      model_path: 保存的生成器模型权重文件路径（例如：best_generator_epoch_XX.pth）
      output_dir: 评估结果保存目录，预测图像、目标图像以及最佳结果图像将保存在此目录下
      input_size: 模型输入尺寸（与训练时一致）
      target_size: 模型目标图像尺寸（与训练时一致）
      upscale_factor: 放大倍数
      in_depth: 输入数据通道数
      cvars: 条件变量列表
      log_transform: 是否对数据做 log 反变换（与训练时保持一致）
      spatial_attention, channel_attention: 注意力机制参数
      model_size: 模型大小（用于确定各阶段通道数），例如训练时设置为 256
      batch_size: 批量大小，评估时建议设为 1
      device: 设备（cpu 或 cuda），若为 None 则自动检测
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构造测试数据集（与训练时使用的 PrecipitationDatasetCGAN 相同）
    test_dataset = PrecipitationDatasetCGAN(
        data_dir=test_data_dir,
        input_size=input_size,
        target_size=target_size,
        upscale_factor=upscale_factor,
        in_depth=in_depth,
        cvars=cvars,
        log_transform=log_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 根据训练时的参数构造生成器模型
    generator = EncodedGenerator(
        in_ch=in_depth,
        ncvar=len(cvars),
        use_ele=True,
        cam=channel_attention,
        sam=spatial_attention,
        stage_chs=[model_size // (2 ** d) for d in range(4)]
    )
    generator = generator.to(device)

    # 加载训练过程中保存的最佳生成器权重
    # 注意：如果训练时使用了 DataParallel，加载时可能需要去掉 "module." 前缀
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    generator.load_state_dict(new_state_dict)
    generator.eval()

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 初始化指标累计变量
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_rmse = 0.0
    total_cc = 0.0
    total_pc, total_po, total_far = 0.0, 0.0, 0.0
    count = 0

    best_psnr = -float('inf')
    best_pred_np = None
    best_hr_np = None

    print("开始评估 encoded-cGAN 模型...")
    with torch.no_grad():
        for idx, (lr_input, cvars_input, hr_target, hr_ele) in enumerate(tqdm(test_loader, desc="Evaluating")):
            # 将各输入移动到 device 上（注意：cvars_input 为列表）
            lr_input = lr_input.to(device)
            cvars_input = [cv.to(device) for cv in cvars_input]
            hr_target = hr_target.to(device)
            hr_ele = hr_ele.to(device)

            # 模型预测
            pred = generator(lr_input, cvars_input, elevation=hr_ele)

            # 转换为 numpy 数组（一般形状为 [1, C, H, W]）
            pred_np = pred.cpu().numpy()
            hr_np = hr_target.cpu().numpy()

            # 如训练时对数据做了 log 变换，此处反变换
            if log_transform:
                pred_np = np.exp(pred_np) - 1
                hr_np = np.exp(hr_np) - 1

            # 计算 MSE
            mse = np.mean((hr_np - pred_np) ** 2)
            # 计算 PSNR（动态范围采用目标图像的极差，防止极差为 0）
            data_range = hr_np.max() - hr_np.min() if hr_np.max() != hr_np.min() else 1.0
            psnr_value = calculate_psnr(hr_np, pred_np, data_range=data_range)
            # 计算 SSIM：这里默认评估第一个样本、第一通道
            ssim_value = compare_ssim(hr_np[0, 0], pred_np[0, 0], data_range=data_range)
            # 计算 RMSE
            rmse_value = np.sqrt(np.mean((pred_np - hr_np) ** 2))
            # 计算皮尔逊相关系数
            cc_value = pearsonr(hr_np.flatten(), pred_np.flatten())[0]

            pc, po, far = calculate_metrics(pred_np, hr_np)

            total_mse += mse
            total_psnr += psnr_value
            total_ssim += ssim_value
            total_rmse += rmse_value
            total_cc += cc_value
            total_pc += pc
            total_po += po
            total_far += far
            count += 1

            # 保存当前样本的预测与目标图像
            pred_img_path = os.path.join(output_dir, f"prediction_{idx+1}.png")
            hr_img_path = os.path.join(output_dir, f"target_{idx+1}.png")
            save_image(pred_np[0, 0], pred_img_path)
            save_image(hr_np[0, 0], hr_img_path)

            # 记录 PSNR 最佳的结果
            if psnr_value > best_psnr:
                best_psnr = psnr_value
                best_pred_np = pred_np
                best_hr_np = hr_np

    # 计算平均指标
    avg_mse = total_mse / count
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_rmse = total_rmse / count
    avg_pc = total_pc / count
    avg_po = total_po / count
    avg_far = total_far / count
    avg_cc = total_cc / count

    print(f"\n评估完成，共评估 {count} 个样本：")
    print(f"平均 PSNR: {avg_psnr:.2f}, 平均 SSIM: {avg_ssim:.4f}, 平均 RMSE: {avg_rmse:.4f}")
    print(f"平均 PC: {avg_pc:.2f}, 平均 PO: {avg_po:.4f}, 平均 FAR: {avg_far:.4f}")
    print(f"平均皮尔逊相关系数: {avg_cc:.4f}")
    print(f"最佳 PSNR: {best_psnr:.2f}")

    # 保存最佳结果图像
    if best_pred_np is not None and best_hr_np is not None:
        best_pred_path = os.path.join(output_dir, "best_prediction.png")
        best_hr_path = os.path.join(output_dir, "best_target.png")
        save_image(best_pred_np[0, 0], best_pred_path)
        save_image(best_hr_np[0, 0], best_hr_path)

    return avg_mse, avg_psnr, avg_ssim, avg_cc

if __name__ == "__main__":
    # ========== 参数配置 ==========
    # 测试数据目录（根据实际情况修改）
    test_data_dir = "../../processed_datasets/new_test/"
    # 生成器模型权重文件路径（注意修改为实际保存的 checkpoint 文件）
    model_path = "../../models/encoded_cgan_output/best_generator_epoch_82.pth"
    # 评估结果保存目录
    output_dir = "../../predictions/encoded_cgan_eval/"

    # 训练/评估时用到的参数（与 encoded_cgan_main 中的设置保持一致）
    input_size = (28, 28)
    target_size = (224, 224)
    upscale_factor = 8
    in_depth = 1
    cvars = ['r2', 't', 'u10', 'v10', 'lsm']
    log_transform = False
    spatial_attention = True
    channel_attention = True
    model_size = 256
    batch_size = 1  # 评估时逐个样本预测

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== 开始评估 ==========
    evaluate_encoded_cgan(test_data_dir, model_path, output_dir,
                          input_size=input_size, target_size=target_size,
                          upscale_factor=upscale_factor, in_depth=in_depth, cvars=cvars,
                          log_transform=log_transform, spatial_attention=spatial_attention,
                          channel_attention=channel_attention, model_size=model_size,
                          batch_size=batch_size, device=device)