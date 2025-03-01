import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr

# =============== 1. 定义多阈值的二值化指标函数 ===============
def calculate_binary_metrics(pred, target, threshold):
    """
    计算给定threshold下的各种二值化降水预测指标:
      PC  : 晴雨准确率 (%)
      PO  : 漏报率    (%)
      FAR : 空报率    (%)
      POD : 命中率    (%)
      CSI : 临界成功指数 (0~1)
      FB  : 频率偏差
      HSS : Heidke Skill Score
    返回一个字典。
    """
    pred_binary = (pred > threshold).astype(int)
    target_binary = (target > threshold).astype(int)

    # 四格表统计
    NA = ((pred_binary == 1) & (target_binary == 1)).sum()  # 命中
    NB = ((pred_binary == 1) & (target_binary == 0)).sum()  # 空报
    NC = ((pred_binary == 0) & (target_binary == 1)).sum()  # 漏报
    ND = ((pred_binary == 0) & (target_binary == 0)).sum()  # 正确无雨

    total = NA + NB + NC + ND

    # PC (晴雨准确率, Percent Correct)
    PC  = 100.0 * (NA + ND) / total if total > 0 else 0.0
    # PO (漏报率, Probability of Miss)
    PO  = 100.0 * NC / (NA + NC) if (NA + NC) > 0 else 0.0
    # FAR (空报率, False Alarm Ratio)
    FAR = 100.0 * NB / (NA + NB) if (NA + NB) > 0 else 0.0
    # POD (命中率, Probability of Detection)
    POD = 100.0 * NA / (NA + NC) if (NA + NC) > 0 else 0.0

    # CSI (临界成功指数, Critical Success Index)
    denom_csi = NA + NB + NC
    CSI = NA / denom_csi if denom_csi > 0 else 0.0

    # FB (频率偏差, Frequency Bias)
    denom_fb = (NA + NC)
    FB = (NA + NB) / denom_fb if denom_fb > 0 else 0.0

    # HSS (Heidke Skill Score)
    numerator = 2.0 * (NA * ND - NB * NC)
    denominator = (NA + NC) * (NB + ND) + (NA + NB) * (NC + ND)
    HSS = numerator / denominator if denominator != 0 else 0.0

    return {
        "PC":  PC,
        "PO":  PO,
        "FAR": FAR,
        "POD": POD,
        "CSI": CSI,
        "FB":  FB,
        "HSS": HSS
    }

def evaluate_multiple_thresholds(pred, target, thresholds):
    """
    给定预测图和真实图，在多个阈值下计算二值化指标。
    返回:
      {
         threshold1: { 'PC': x, 'PO': x, 'FAR': x, ... },
         threshold2: { 'PC': x, 'PO': x, 'FAR': x, ... },
         ...
      }
    """
    results = {}
    for th in thresholds:
        metrics_dict = calculate_binary_metrics(pred, target, th)
        results[th] = metrics_dict
    return results

# =============== 2. 定义对单个样本的综合评估函数 ===============
def evaluate_single_sample(pred_img, target_img, thresholds):
    """
    1) 计算PSNR, SSIM, RMSE等通用图像指标
    2) 在多个阈值下计算降水预测的二值化指标

    返回:
      psnr_value, ssim_value, rmse_value, multi_threshold_metrics
    其中 multi_threshold_metrics 是一个 dict, key=threshold, value=指标dict
    """
    # 通用像素指标
    data_range = target_img.max() - target_img.min()
    data_range = max(data_range, 1e-6)  # 防止分母为0
    psnr_value = calculate_psnr(target_img, pred_img, data_range=data_range)
    ssim_value = compare_ssim(target_img, pred_img, data_range=data_range)
    rmse_value = np.sqrt(np.mean((pred_img - target_img) ** 2))

    # 多阈值二值化指标
    multi_threshold_metrics = evaluate_multiple_thresholds(pred_img, target_img, thresholds)

    return psnr_value, ssim_value, rmse_value, multi_threshold_metrics

# =============== 3. 主评估流程 ===============
def main_evaluation(path):
    # (1) 读取 netCDF 文件
    nc_file = os.path.join(path, "results/samples.nc")
    ds = xr.open_dataset(nc_file)

    # 获取数据
    coarse = ds["coarse"].values     # shape: [time, lat, lon]
    fine = ds["fine"].values         # shape: [time, lat, lon]
    predicted = ds["predicted"].values  # shape: [time, lat, lon]

    # 将 predicted 中的负值置为 0
    predicted[predicted < 0] = 0

    # 时间维度大小
    ntime = coarse.shape[0]

    # (2) 设置保存图像的文件夹
    save_dir = os.path.join(path, "results/combined_images_origin/")
    os.makedirs(save_dir, exist_ok=True)

    # (3) 定义多阈值
    thresholds = [0.1, 1, 10, 20, 50, 100]

    # 初始化累加器: PSNR/SSIM/RMSE
    total_psnr = 0.0
    total_ssim = 0.0
    total_rmse = 0.0
    num_samples = 0

    # 二值化指标累加器 (各阈值分别累加)
    # threshold_metrics_sum = {
    #   0.1: {'PC':0, 'PO':0, 'FAR':0, 'POD':0, 'CSI':0, 'FB':0, 'HSS':0, 'count':0},
    #   1.0:  {...},
    #   ...
    # }
    threshold_metrics_sum = {}
    for th in thresholds:
        threshold_metrics_sum[th] = {
            "PC":0, "PO":0, "FAR":0, "POD":0, "CSI":0, "FB":0, "HSS":0, "count":0
        }

    # (4) 逐时间步评估
    for t in tqdm(range(ntime)):
        coarse_img = coarse[t]
        fine_img   = fine[t]
        pred_img   = predicted[t]

        # (4.1) 作图并保存
        data_min = min(coarse_img.min(), fine_img.min(), pred_img.min())
        data_max = max(coarse_img.max(), fine_img.max(), pred_img.max())

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Coarse
        ax = axes[0]
        c = ax.imshow(coarse_img, cmap='viridis', vmin=data_min, vmax=data_max)
        ax.set_title(f"Coarse {t+1}")
        fig.colorbar(c, ax=ax)

        # Fine
        ax = axes[1]
        c = ax.imshow(fine_img, cmap='viridis', vmin=data_min, vmax=data_max)
        ax.set_title(f"Fine {t+1}")
        fig.colorbar(c, ax=ax)

        # Predicted
        ax = axes[2]
        c = ax.imshow(pred_img, cmap='viridis', vmin=data_min, vmax=data_max)
        ax.set_title(f"Predicted {t+1}")
        fig.colorbar(c, ax=ax)

        combined_image_path = os.path.join(save_dir, f"combined_{t+1:04d}.png")
        plt.tight_layout()
        plt.savefig(combined_image_path)
        plt.close()

        # (4.2) 计算评估指标
        psnr_val, ssim_val, rmse_val, mth_metrics = evaluate_single_sample(
            pred_img, fine_img, thresholds
        )

        # 累加像素指标
        total_psnr += psnr_val
        total_ssim += ssim_val
        total_rmse += rmse_val
        num_samples += 1

        # 累加多阈值二值化指标
        for th in thresholds:
            for metric_key in ["PC","PO","FAR","POD","CSI","FB","HSS"]:
                threshold_metrics_sum[th][metric_key] += mth_metrics[th][metric_key]
            threshold_metrics_sum[th]["count"] += 1

    # (5) 计算并打印最终平均结果
    print(f"\nAverage pixel-level metrics over {num_samples} samples:")
    print(f"PSNR: {total_psnr / num_samples:.2f}")
    print(f"SSIM: {total_ssim / num_samples:.4f}")
    print(f"RMSE: {total_rmse / num_samples:.4f}")

    print("\n=== Multi-threshold metrics ===")
    for th in thresholds:
        c = threshold_metrics_sum[th]["count"]
        if c == 0:
            continue
        avg_PC  = threshold_metrics_sum[th]["PC"]  / c
        avg_PO  = threshold_metrics_sum[th]["PO"]  / c
        avg_FAR = threshold_metrics_sum[th]["FAR"] / c
        avg_POD = threshold_metrics_sum[th]["POD"] / c
        avg_CSI = threshold_metrics_sum[th]["CSI"] / c
        avg_FB  = threshold_metrics_sum[th]["FB"]  / c
        avg_HSS = threshold_metrics_sum[th]["HSS"] / c

        print(f"\n[Threshold = {th} mm]")
        print(f"  PC :  {avg_PC:.2f}%")
        print(f"  PO :  {avg_PO:.2f}%")
        print(f"  FAR:  {avg_FAR:.2f}%")
        print(f"  POD:  {avg_POD:.2f}%")
        print(f"  CSI:  {avg_CSI:.3f}")
        print(f"  FB :  {avg_FB:.3f}")
        print(f"  HSS:  {avg_HSS:.3f}")

    print("\nAll images saved and metrics computed successfully.")

if __name__ == "__main__":
    path = "../../predictions/diffusion_unet"
    main_evaluation(path)
