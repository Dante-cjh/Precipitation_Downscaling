import os
import torch
from torch.utils.data import DataLoader
from src.dataset.ddpm_dataset import RobustPrecipitationZScoreDataset


def test_residual_to_fine_image(data_dir):
    # 创建数据集实例
    dataset = RobustPrecipitationZScoreDataset(data_dir)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # 遍历数据集中的每个样本
    for i, batch in enumerate(dataloader):
        inputs = batch["inputs"]
        targets = batch["targets"]
        fine_acpcp = batch["fine_acpcp"]
        coarse_acpcp = batch["coarse_acpcp"]

        # 使用 residual_to_fine_image 方法
        predicted_fine = dataset.residual_to_fine_image(targets, coarse_acpcp)

        # 检查 predicted_fine 和 fine_acpcp 是否一致
        if not torch.allclose(predicted_fine, fine_acpcp, atol=1e-6):
            print(f"Sample {i} failed: predicted_fine and fine_acpcp are not close.")
            print(f"predicted_fine: {predicted_fine}")
            print(f"fine_acpcp: {fine_acpcp}")

    print("Test completed.")


if __name__ == "__main__":
    data_dir = "../processed_datasets/new_test/"
    test_residual_to_fine_image(data_dir)