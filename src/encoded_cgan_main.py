import os
import shutil
import torch
from encoded_cgan_train import train
from dataset.encoded_cgan_dataset import PrecipitationDatasetCGAN

def main():
    # 初始化数据集
    train_dataset = PrecipitationDatasetCGAN(data_dir=train_data_dir, input_size=(28, 28), target_size=(224, 224),
                                             upscale_factor=upscale_factor, in_depth=input_depth, cvars=cvars,
                                             log_transform=log_transform)

    val_dataset = PrecipitationDatasetCGAN(data_dir=val_data_dir, input_size=(28, 28), target_size=(224, 224),
                                           upscale_factor=upscale_factor, in_depth=input_depth, cvars=cvars,
                                           log_transform=log_transform)

    # 调用训练函数
    train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        warmup_epochs=warmup_epochs,
        model_size=model_size,
        upscale_factor=upscale_factor,
        cvars=cvars,
        log_transform=log_transform,
        spatial_attention=spatial_attention,
        channel_attention=channel_attention,
        weight_mse=weight_mse,
        output_dir=output_dir,
        results_dir=results_dir,
        log_dir="../logs/encoded_cgan/"
    )


if __name__ == "__main__":

    # 定义全局变量
    batch_size = 64
    learning_rate = 3e-4
    max_epochs = 100
    warmup_epochs = 100
    input_depth = 1
    model_size = 256
    upscale_factor = 8
    log_transform = False
    spatial_attention = True
    channel_attention = True
    weight_mse = 5
    cvars = ['r2', 't', 'u10', 'v10', 'lsm']

    # 数据路径
    train_data_dir = "../processed_datasets/new_train/"
    val_data_dir = "../processed_datasets/new_val/"
    output_dir = "../models/encoded_cgan_output/"
    results_dir = "../predictions/encoded_cgan_best/"

    # 创建输出目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


    # 调用主函数
    main()
