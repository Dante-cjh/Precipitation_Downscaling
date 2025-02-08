import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset.v4_dataset import PrecipitationDatasetV4
from src.models.srcnn import SRCNN
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm  # 进度条库

from eval import evaluate_model
from skimage.metrics import peak_signal_noise_ratio as psnr


# PSNR 计算函数
def calculate_psnr(pred, target, data_range=None):
    """
    使用 skimage 的 psnr 函数计算 PSNR。

    参数:
    - pred: 预测图像（numpy.ndarray 或 torch.Tensor）
    - target: 目标图像（numpy.ndarray 或 torch.Tensor）
    - data_range: 图像的动态范围（通常为 1.0 或 255）

    返回:
    - PSNR 值（float）
    """
    # 如果输入是 torch.Tensor，将其转换为 numpy.ndarray
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # 确保数据维度匹配
    if pred.shape != target.shape:
        raise ValueError("预测图像和目标图像的形状必须一致")

    # 如果未指定 data_range，动态计算
    if data_range is None:
        data_range = target.max() - target.min()

    # 计算 PSNR
    return psnr(target, pred, data_range=data_range)


# RMSE 计算函数
def calculate_rmse(pred, target):
    mse = torch.mean((pred - target) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        """
        初始化早停机制
        参数:
            patience (int): 在验证集性能不提升时，最多允许的连续轮次
            min_delta (float): 性能提升的最小变化
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        """
        更新早停机制
        参数:
            val_loss (float): 当前轮次的验证损失
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # 重置计数器
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# 训练函数
def train_model(train_dir, val_dir, input_size, target_size, upscale_factor, model_save_path, batch_size, epochs, device, log_dir, patience=10):
    # 数据加载
    train_dataset = PrecipitationDatasetV4(train_dir, input_size, target_size, upscale_factor)
    val_dataset = PrecipitationDatasetV4(val_dir, input_size, target_size, upscale_factor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = SRCNN(input_channels=1).to(device)
    criterion = nn.MSELoss()

    # 设置优化器
    optimizer = optim.Adam([
        {'params': model.conv1.parameters(), 'lr': 1e-4},
        {'params': model.conv2.parameters(), 'lr': 1e-4},
        {'params': model.conv3.parameters(), 'lr': 1e-5},
    ])

    # TensorboardX
    writer = SummaryWriter(log_dir)

    # 早停机制
    early_stopping = EarlyStopping(patience=patience)

    print("开始训练...")
    for epoch in range(epochs):
        # 初始化进度条
        train_progress = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}] - Training", unit="batch")
        model.train()
        train_loss, train_psnr, train_rmse = 0.0, 0.0, 0.0

        for lr, hr in train_progress:
            lr, hr = lr.to(device), hr.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(lr)
            loss = criterion(outputs, hr)
            loss.backward()
            optimizer.step()

            # 更新指标
            train_loss += loss.item()
            train_psnr += calculate_psnr(outputs, hr)
            train_rmse = calculate_rmse(outputs, hr)

            # 显示当前 batch 的损失
            train_progress.set_postfix(loss=loss.item())

        # 计算 epoch 平均指标
        train_loss /= len(train_loader)
        train_psnr /= len(train_loader)
        train_rmse /= len(train_loader)

        # 验证阶段
        model.eval()
        val_loss, val_psnr, val_rmse = 0.0, 0.0, 0.0
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{epochs}] - Validation", unit="batch")
            for lr, hr in val_progress:
                lr, hr = lr.to(device), hr.to(device)
                outputs = model(lr)
                loss = criterion(outputs, hr)
                val_loss += loss.item()
                val_psnr += calculate_psnr(outputs, hr)
                val_rmse += calculate_rmse(outputs, hr)

        # 计算 epoch 平均指标
        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        val_rmse /= len(val_loader)

        # 打印日志
        print(f"\nEpoch [{epoch + 1}/{epochs}] Completed")
        print(f"Train Loss: {train_loss:.6f}, Train PSNR: {train_psnr:.2f}, Train RMSE: {train_rmse:.4f}")
        print(f"Val Loss: {val_loss:.6f}, Val PSNR: {val_psnr:.2f}, Val RMSE: {val_rmse:.4f}")

        # 记录到 Tensorboard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('PSNR/Train', train_psnr, epoch)
        writer.add_scalar('PSNR/Validation', val_psnr, epoch)
        writer.add_scalar('RMSE/Train', train_rmse, epoch)
        writer.add_scalar('RMSE/Validation', val_rmse, epoch)

        # 检查早停条件
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("验证损失未改善，触发早停机制")
            break

    # 保存模型
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存至: {model_save_path}")


def train_stacked_model(layers, train_dir, val_dir, test_dir, output_dir, model_save_path, batch_size, epochs, device, log_dir):
    """
    训练堆叠的 SRCNN 模型，每一层独立训练，逐步提高分辨率。

    参数:
        upscale_factor: 每层的放大倍数
        layers: 堆叠的层数
        train_dir: 训练数据的文件夹
        val_dir: 验证数据的文件夹
        model_save_path: 模型保存的路径
        upscale_factor: 每层的放大倍数（例如 2 表示 2x 放大）
        batch_size: 批量大小
        epochs: 每层的训练轮数
        device: 训练设备（CPU/GPU/MPS）
        log_dir: TensorBoard 日志保存路径
    """
    # 堆叠参数
    resolutions = [(28, 28), (56, 56), (112, 112), (224, 224)]  # 逐层分辨率
    upscale_factor = 2  # 每次 2x 放大

    for layer in range(layers):
        input_size = resolutions[layer]
        target_size = resolutions[layer + 1]

        print(f"\n开始训练第 {layer + 1} 层模型：输入大小 {input_size} -> 输出大小 {target_size}")

        # 为每一层创建独立的模型保存路径
        model_save_path_dir = f"{model_save_path}_layer{layer + 1}.pth"

        # 为每一层创建独立的日志目录
        layer_log_dir = os.path.join(log_dir, f"layer_{layer + 1}")
        os.makedirs(layer_log_dir, exist_ok=True)

        # 训练当前层
        train_model(train_dir, val_dir, input_size, target_size, upscale_factor, model_save_path_dir,
                    batch_size, epochs, device, log_dir=layer_log_dir)

        # 评估当前层
        output_dir_dir = os.path.join(output_dir, f"predictions_layer_{layer + 1}")
        os.makedirs(output_dir_dir, exist_ok=True)

        print(f"\n评估第 {layer + 1} 层模型性能...")
        evaluate_model(test_dir, input_size, target_size, model_save_path_dir, upscale_factor, output_dir=output_dir_dir, device=device)


    print("\n所有层的训练完成！")