import torch
from torch.utils.data import DataLoader
from models.encoded_cgan import EncodedGenerator
from dataset.encoded_cgan_dataset import PrecipitationDatasetCGAN
from encoded_cgan_train import validate
from torch.utils.tensorboard import SummaryWriter

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化数据集
val_data_dir = "../../processed_datasets/new_val/"
val_dataset = PrecipitationDatasetCGAN(data_dir=val_data_dir, input_size=(28, 28), target_size=(224, 224),
                                       upscale_factor=8, in_depth=1, cvars=['r2', 't', 'u10', 'v10', 'lsm'],
                                       log_transform=False)

# 初始化模型
model_size = 256
cvars = ['r2', 't', 'u10', 'v10', 'lsm']
generator = EncodedGenerator(in_ch=1, ncvar=len(cvars), use_ele=True, cam=True, sam=True,
                             stage_chs=[model_size // 2 ** d for d in range(4)])
generator = generator.to(device)

# 加载预训练模型权重
generator.load_state_dict(torch.load("../models/encoded_cgan_output/generator_epoch_100.pth"))

# 初始化 Tensorboard
writer = SummaryWriter(log_dir="../../logs/encoded_cgan/")

# 调用验证函数
validate(generator, val_dataset, device, epoch=100, log_transform=False, output_dir="../../models/encoded_cgan_output/", writer=writer)

# 关闭 Tensorboard
writer.close()