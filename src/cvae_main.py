from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from cvae_train import cVAETrainer
from dataset.cvae_dateset import PrecipitationDataModule


def main():
    # Set random seed for reproducibility
    seed_everything(42)

    # Paths
    train_path = '../processed_datasets/new_train/'
    val_path = '../processed_datasets/new_val/'
    test_path = '../processed_datasets/new_test/'
    log_dir = '../logs/'
    models_dir = '../models/cvae/'

    # Hyperparameters
    input_size = (28, 28)
    target_size = (224, 224)
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 1000
    patience = 30

    # Data module
    data_module = PrecipitationDataModule(train_path, val_path, test_path, input_size, target_size, batch_size, normalize=True)

    # Initialize model
    model = cVAETrainer(spatial_x_dim=28 * 28, out_dim=224 * 224, learning_rate=learning_rate)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=models_dir, save_top_k=1, monitor='val_loss', mode='min')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Logger
    logger = TensorBoardLogger(log_dir, name='cvae')

    # Trainer
    trainer = Trainer(max_epochs=num_epochs,
                      callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
                      accelerator="auto",
                      logger=logger)

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Evaluate the model
    trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    main()