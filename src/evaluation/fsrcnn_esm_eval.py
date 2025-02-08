import heapq
import os
import sys
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.dataset.fsrcnn_dataset import PrecipitationDataset
from src.fsrcnn_train import FSRCNNTrainer


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

def evaluate_fsrcnn(test_dir, model_path, output_dir, device):
    if not os.path.exists(test_dir):
        print(f"Error: The directory {test_dir} does not exist.")
        sys.exit(1)

    test_dataset = PrecipitationDataset(test_dir, (28, 28), (224, 224))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = FSRCNNTrainer.load_from_checkpoint(model_path, scale_factor=8, learning_rate=0.001)
    model = model.to(device)
    model.eval()

    sr_output_dir = os.path.join(output_dir, "results", "SR")
    hr_output_dir = os.path.join(output_dir, "results", "HR")
    os.makedirs(sr_output_dir, exist_ok=True)
    os.makedirs(hr_output_dir, exist_ok=True)

    total_psnr, total_ssim, total_rmse, total_pc, total_po, total_far = 0, 0, 0, 0, 0, 0
    count = 0
    min_rmse_heap = []

    print("\n开始评估 FSRCNN 模型...")
    with torch.no_grad():
        for idx, (lr, hr) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
            hr = hr.to(device)
            lr = lr.to(device)
            sr = model(lr)

            sr_np = sr.cpu().numpy().squeeze()
            hr_np = hr.cpu().numpy().squeeze()

            psnr_value = calculate_psnr(sr_np, hr_np, data_range=hr_np.max() - hr_np.min())
            ssim_value = compare_ssim(sr_np, hr_np, data_range=hr_np.max() - hr_np.min())
            rmse_value = np.sqrt(np.mean((sr_np - hr_np) ** 2))
            pc, po, far = calculate_metrics(sr_np, hr_np)

            heapq.heappush(min_rmse_heap, (rmse_value, f"prediction_{idx + 1}.png"))
            if len(min_rmse_heap) > 5:
                heapq.heappop(min_rmse_heap)

            total_psnr += psnr_value
            total_ssim += ssim_value
            total_rmse += rmse_value
            total_pc += pc
            total_po += po
            total_far += far
            count += 1

            sr_img_path = os.path.join(sr_output_dir, f"prediction_{idx + 1}.png")
            hr_img_path = os.path.join(hr_output_dir, f"target_{idx + 1}.png")

            sr_img = _normalize_to_8bit(sr_np)
            cv2.imwrite(sr_img_path, sr_img)

            hr_img = _normalize_to_8bit(hr_np)
            cv2.imwrite(hr_img_path, hr_img)

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_rmse = total_rmse / count
    avg_pc = total_pc / count
    avg_po = total_po / count
    avg_far = total_far / count

    min_rmse_images = sorted(min_rmse_heap)

    print(f"\n评估完成！")
    print(f"平均 PSNR: {avg_psnr:.2f}, 平均 SSIM: {avg_ssim:.4f}, 平均 RMSE: {avg_rmse:.4f}")
    print(f"平均 PC: {avg_pc:.2f}%, 平均 PO: {avg_po:.2f}%, 平均 FAR: {avg_far:.2f}%")
    print(f"Top 5 images with smallest RMSE:")
    for rmse, image_name in min_rmse_images:
        print(f"Image: {image_name}, RMSE: {rmse:.4f}")

    return avg_psnr, avg_ssim, avg_rmse, avg_pc, avg_po, avg_far, min_rmse_images

def _normalize_to_8bit(array):
    arr_min, arr_max = array.min(), array.max()
    if np.isclose(arr_min, arr_max):
        return np.zeros_like(array, dtype=np.uint8)
    norm = (array - arr_min) / (arr_max - arr_min)
    norm_255 = (norm * 255).astype(np.uint8)
    return norm_255

if __name__ == "__main__":
    TRAIN_PATH = "../../processed_datasets/new_train"
    VAL_PATH = "../../processed_datasets/new_val"
    TEST_PATH = "../../processed_datasets/new_test"
    MODEL_PATH = "../../models/fsrcnn_esm/fsrcnn_esm_epochepoch=69_val_lossval_loss=0.0001.ckpt"
    OUTPUT_DIR = "../../predictions/fsrcnn_esm"
    LOG_DIR = "../../logs/fsrcnn_esm"

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')


    evaluate_fsrcnn(TEST_PATH, MODEL_PATH, OUTPUT_DIR, DEVICE)