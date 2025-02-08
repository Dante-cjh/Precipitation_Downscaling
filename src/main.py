from train import train_stacked_model
import torch

if __name__ == '__main__':
    # 配置
    TRAIN_DIR = '../processed_datasets/new_train'
    VAL_DIR = '../processed_datasets/new_val'
    TEST_DIR = '../processed_datasets/new_test'
    MODEL_SAVE_PATH = '../models/v4/srcnn'
    OUTPUT_DIR = '../predictions/v4'
    LOG_DIR = '../logs/v4'

    UPSCALE_FACTOR = 2  # 每次 2x 放大，三次累计就 8x 放大
    STACKED_LAYERS = 3  # 堆叠 3 层
    BATCH_SIZE = 200  # 批次大小
    EPOCHS = 1  # 每层的 epoch 数

    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # 堆叠训练
    train_stacked_model(STACKED_LAYERS ,TRAIN_DIR, VAL_DIR, TEST_DIR, OUTPUT_DIR, MODEL_SAVE_PATH, BATCH_SIZE, EPOCHS, device, LOG_DIR)
