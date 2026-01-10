#Model based on NVIDIA's DAVE-2 and Conditional Imitation Learning with constant throttle=0.18 and brake=0.0
import pandas as pd
import torch
import cv2
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
from pathlib import Path

class DrivingDataset(Dataset):
    def __init__(self, csv_file, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations.iloc[idx]['image']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        if self.transform:
            image = self.transform(image)
        steer = self.annotations.iloc[idx]['steer']
        #"-1" because indexes are 0-3 end commands are 1-4
        command = torch.tensor(int(self.annotations.iloc[idx]['command'] - 1), dtype = torch.long)
        return image, command, steer

class DrivingDataModule (pl.LightningDataModule):
    def __init__(self, csv_path, batch_size = 32):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size

    def setup(self, stage = None):
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((66,200)), transforms.ToTensor()])
        full_dataset = DrivingDataset(self.csv_path, transform)
        self.val_dataset, self.train_dataset = random_split(full_dataset,
                                                             [len(full_dataset)//10, len(full_dataset) - len(full_dataset)//10])
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

from torchmetrics.regression import MeanAbsoluteError
from torch import nn
from torch import optim
import torch.nn.functional as F

class DrivingModel(pl.LightningModule):
    def __init__(self, learning_rate = 1e-4):
        super().__init__()
        self.loss_function = nn.MSELoss()
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.learning_rate = learning_rate

        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ELU(),
            nn.Flatten(),
        )

        self.control_branches = nn.ModuleList([
            nn.Sequential(
                nn.LazyLinear(100),
                nn.Dropout(0.35),
                nn.ELU(),
                nn.Linear(100, 50),
                nn.Dropout(0.35),
                nn.ELU(),
                nn.Linear(50, 10),
                nn.Dropout(0.35),
                nn.ELU(),
                nn.Linear(10, 1)
            )
            for i in range(4)
        ])

    def forward(self, x):
        x = self.conv_block(x)
        pred_control = torch.cat([control(x) for control in self.control_branches], dim = 1)

        return pred_control

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        image, command, steer = train_batch

        outputs = self.forward(image.float())
        steer = steer.view(-1, 1).float()
        command = command.view(-1, 1).long()

        selected_outputs = torch.gather(outputs, 1, command)

        loss = self.loss_function(selected_outputs, steer)

        self.log('train_loss', loss, on_step= True, on_epoch = True)

        mae_value = self.train_mae(selected_outputs, steer)
        self.log('train_mae', mae_value, on_epoch=True, on_step= False)

        return loss

    def validation_step(self, train_batch, batch_idx):
        image, command, steer = train_batch

        outputs = self.forward(image.float())
        steer = steer.view(-1, 1).float()
        command = command.view(-1, 1).long()

        selected_outputs = torch.gather(outputs, 1, command)

        loss = self.loss_function(selected_outputs, steer)

        self.log('val_loss', loss, on_step= True, on_epoch = True)

        mae_value = self.train_mae(selected_outputs, steer)
        self.log('val_mae', mae_value, on_epoch=True, on_step= False)

        return loss

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data_path = Path('../../labels/dave2_const_v_s')
    csv_path = data_path / 'final_annotations.csv'

    ddm = DrivingDataModule(str(csv_path))
    ddm.setup()

    driving_model = DrivingModel()

    from pytorch_lightning.loggers import TensorBoardLogger

    #Callaback to resist overfitting
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta = 0.0, patience=3, verbose = True, mode = 'min')
    checkpoint = ModelCheckpoint(monitor = 'val_mae', save_top_k=1, mode = 'min')

    logger=TensorBoardLogger("../../logs", name="agent_dave2_const_v_b_CIL")
    trainer=pl.Trainer(logger=logger, max_epochs=30, log_every_n_steps=1, accelerator="gpu", callbacks = [early_stop_callback, checkpoint])
    trainer.fit(driving_model, ddm)