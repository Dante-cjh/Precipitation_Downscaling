from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from fsrcnn_train import FSRCNNTrainer
from dataset.fsrcnn_dataset import PrecipitationDataModule

if __name__ == "__main__":
    # 配置路径
    TRAIN_PATH = "../processed_datasets/new_train"
    VAL_PATH = "../processed_datasets/new_val"
    MODEL_DIR = "../models/fsrcnn_esm"
    LOG_DIR = "../logs/fsrcnn_esm"

    # 参数设置
    INPUT_SIZE = (28, 28)
    TARGET_SIZE = (224, 224)
    SCALE_FACTOR = 8
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    MAX_EPOCHS = 100

    # 数据模块
    data_module = PrecipitationDataModule(TRAIN_PATH, VAL_PATH, INPUT_SIZE, TARGET_SIZE, BATCH_SIZE)

    # 模型
    model = FSRCNNTrainer(scale_factor=SCALE_FACTOR, learning_rate=LEARNING_RATE)

    # 保存模型权重的回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_DIR,  # 模型保存路径
        filename="fsrcnn_esm_epoch{epoch:02d}_val_loss{val_loss:.4f}",  # 文件命名规则
        save_top_k=1,  # 仅保存最优模型
        monitor="val_loss",  # 监控验证损失
        mode="min"  # 验证损失越小越好
    )

    # 训练器
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5), checkpoint_callback],
        log_every_n_steps=1,
        accelerator="auto",
        default_root_dir=LOG_DIR  # 日志保存路径
    )

    # 开始训练
    trainer.fit(model, data_module)

    print(f"最佳模型保存在: {checkpoint_callback.best_model_path}")