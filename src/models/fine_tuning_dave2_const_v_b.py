from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from model_dave2_const_v_b import DrivingModel, DrivingDataset, DrivingDataModule
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch
from torchinfo import summary

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    log_path = Path("../../logs")
    agent_path = Path("agent_dave2_const_v_b")
    #Version to fine tune - by far 22 the best
    version = 22

    checkpoint_path = log_path / agent_path / Path(f"version_{str(version)}/checkpoints")

    try:
        model_path = next(checkpoint_path.glob("*ckpt"))
    except StopIteration:
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")

    #Lodaing model to gpu
    model = DrivingModel.load_from_checkpoint(model_path, train_flag = True, version = version, learning_rate = 1e-5, fine_tune_flag = True)

    # summary(model, input_size=[(1, 3, 66, 200), (1,)])

    for name,param in model.named_parameters():
        if "conv" in name:
            print(f"Froze layer: {name}")
            param.requires_grad = False
        elif "fc" in name:
            param.requires_grad = True


    data_path = Path('../../labels/dave2_const_v_s')
    csv_path = data_path / 'fine_tuning_annotations.csv'

    ddm = DrivingDataModule(str(csv_path), batch_size = 16)

    #Callaback to resist overfitting
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta = 0.0, patience=3, verbose = True, mode = 'min')
    checkpoint = ModelCheckpoint(monitor = 'val_mae', save_top_k=1, mode = 'min')

    logger=TensorBoardLogger("../../logs", name="agent_dave2_const_v_b")
    trainer=pl.Trainer(logger=logger, max_epochs=15, log_every_n_steps=1, accelerator="gpu", callbacks = [early_stop_callback, checkpoint])
    trainer.fit(model, ddm)