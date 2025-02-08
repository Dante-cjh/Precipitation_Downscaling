import os
import cv2
import heapq
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from src.dataset.cvae_dateset import PrecipitationDataset
from src.models.cvae import cVAE
from src.cvae_train import cVAETrainer
from src.util import mat2Dto3D


# 假设你已有的计算二值化指标函数
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


# 将数值图归一化到 8bit 图像
def _normalize_to_8bit(array):
    arr_min, arr_max = array.min(), array.max()
    if np.isclose(arr_min, arr_max):
        return np.zeros_like(array, dtype=np.uint8)
    norm = (array - arr_min) / (arr_max - arr_min)
    norm_255 = (norm * 255).astype(np.uint8)
    return norm_255

def load_cvae_model(model_path, device):
    # 加载 checkpoint 到 CPU
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    # 从 checkpoint 中提取 state_dict
    state_dict = checkpoint["state_dict"]

    # 去除所有键中开头的 "model." 前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_key = key[len("model."):]  # 去掉前缀
        else:
            new_key = key
        new_state_dict[new_key] = value

    # 初始化模型（确保构造参数与训练时一致）
    model = cVAE(spatial_x_dim=28 * 28, out_dim=224 * 224)
    # 加载新处理的 state_dict
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    return model


def evaluate_cVAE(test_dir, model_path, output_dir, device, threshold=0.1, num_samples=9):
    """
    使用 CVAE 模型进行评估：
      - 对于测试集中的每个样本，生成 num_samples 个预测样本，
      - 计算每个预测与真实图像之间的 PSNR、SSIM、RMSE、PC、PO、FAR 等指标，
      - 选取 RMSE 最低的预测作为当前样本的代表，
      - 保存预测图和真实图，并统计整体指标。
    """
    # 使用示例：
    model = load_cvae_model(MODEL_PATH, device)

    # 构造测试数据集（此处假设输入为低分辨率图像，目标为高分辨率图像；根据需要修改）
    test_dataset = PrecipitationDataset(test_dir, (28, 28), (224, 224))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 准备保存预测图与 HR 图的文件夹
    cvae_output_dir = os.path.join(output_dir, "results", "CVAE")
    hr_output_dir = os.path.join(output_dir, "results", "HR")
    os.makedirs(cvae_output_dir, exist_ok=True)
    os.makedirs(hr_output_dir, exist_ok=True)

    total_psnr, total_ssim, total_rmse = 0, 0, 0
    total_pc, total_po, total_far = 0, 0, 0
    count = 0
    min_rmse_heap = []  # 用于记录 RMSE 最小的前5个预测

    print("\n开始评估 CVAE 模型...")
    with torch.no_grad():
        for idx, (X, Y) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating CVAE"):
            # 将数据放到对应设备上
            X = X.to(device)  # 低分辨率输入
            Y = mat2Dto3D(Y, lat=224, lon=224)
            Y = Y.to(device)  # 高分辨率真实图像

            # 对于当前输入，生成 num_samples 个预测样本
            predictions = []
            for _ in range(num_samples):
                # 假设模型内部已经处理好输入的形状，不需要额外变换
                y_pred = model.predictX(X)  # 输出形状应与 Y 一致，或者为展平后的形式
                predictions.append(y_pred)
            predictions = torch.cat(predictions, dim=0)  # shape: [num_samples, ...]
            predictions_np = predictions.detach().cpu().numpy()
            # 取出真实图像（转换为 numpy 数组，并 squeeze 去除 batch 维度）
            Y_np = Y.cpu().numpy().squeeze()

            # 在生成的多个预测中，选取 RMSE 最小的预测作为当前样本的结果
            best_rmse = float('inf')
            best_prediction = None
            best_metrics = None
            for i in range(predictions_np.shape[0]):
                pred = predictions_np[i]
                # 如果预测输出为一维（展平），则重塑为 (224, 224)
                if pred.ndim == 1:
                    pred_img = pred.reshape(224, 224)
                else:
                    pred_img = pred
                # 计算各项指标
                psnr_value = calculate_psnr(pred_img, Y_np, data_range=Y_np.max() - Y_np.min())
                ssim_value = compare_ssim(pred_img, Y_np, data_range=Y_np.max() - Y_np.min())
                rmse_value = np.sqrt(np.mean((pred_img - Y_np) ** 2))
                pc, po, far = calculate_metrics(pred_img, Y_np, threshold)
                if rmse_value < best_rmse:
                    best_rmse = rmse_value
                    best_prediction = pred_img
                    best_metrics = (psnr_value, ssim_value, rmse_value, pc, po, far)

            # 保存当前样本的预测图和真实图
            sr_img_path = os.path.join(cvae_output_dir, f"prediction_{idx + 1}.png")
            hr_img_path = os.path.join(hr_output_dir, f"target_{idx + 1}.png")
            if Y_np.max() > 0.5:
                print("min:", Y_np.min(), "max:", Y_np.max())
            cv2.imwrite(sr_img_path, _normalize_to_8bit(best_prediction))
            cv2.imwrite(hr_img_path, _normalize_to_8bit(Y_np))

            # 累加指标
            total_psnr += best_metrics[0]
            total_ssim += best_metrics[1]
            total_rmse += best_metrics[2]
            total_pc += best_metrics[3]
            total_po += best_metrics[4]
            total_far += best_metrics[5]
            count += 1

            # 记录 RMSE 最低的样本（用于后续展示）
            heapq.heappush(min_rmse_heap, (best_metrics[2], f"prediction_{idx + 1}.png"))
            if len(min_rmse_heap) > 5:
                heapq.heappop(min_rmse_heap)

    # 计算平均指标
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_rmse = total_rmse / count
    avg_pc = total_pc / count
    avg_po = total_po / count
    avg_far = total_far / count
    min_rmse_images = sorted(min_rmse_heap)

    print(f"\nCVAE 模型评估完成！")
    print(f"平均 PSNR: {avg_psnr:.2f}, 平均 SSIM: {avg_ssim:.4f}, 平均 RMSE: {avg_rmse:.4f}")
    print(f"平均 PC: {avg_pc:.2f}%, 平均 PO: {avg_po:.2f}%, 平均 FAR: {avg_far:.2f}%")
    print(f"RMSE 最小的前 5 张预测图：")
    for rmse, image_name in min_rmse_images:
        print(f"Image: {image_name}, RMSE: {rmse:.4f}")

    return avg_psnr, avg_ssim, avg_rmse, avg_pc, avg_po, avg_far, min_rmse_images


if __name__ == "__main__":
    # 路径设置，根据实际情况修改
    TEST_PATH = "../../processed_datasets/new_test"
    MODEL_PATH = "../../models/cvae/epoch=45-step=5750.ckpt"  # CVAE 模型权重文件
    OUTPUT_DIR = "../../predictions/cVAE"

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 运行评估
    evaluate_cVAE(TEST_PATH, MODEL_PATH, OUTPUT_DIR, device)