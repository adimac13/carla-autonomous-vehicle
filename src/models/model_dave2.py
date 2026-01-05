#Model based on NVIDIA's DAVE-2
#TODO add changing weather
#TODO fix turning constantly left
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
from pathlib import Path

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
        throttle = self.annotations.iloc[idx]['throttle']
        brake = self.annotations.iloc[idx]['brake']
        speed = self.annotations.iloc[idx]['speed']
        labels = torch.tensor([steer, throttle, brake], dtype = torch.float32)

        command = torch.tensor(int(self.annotations.iloc[idx]['command']), dtype = torch.long)

        return image, command, speed, labels

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
    def __init__(self):
        super().__init__()
        self.loss_function = nn.MSELoss()
        self.elu = nn.ELU()
        self.flat = nn.Flatten()
        self.dropout = nn.Dropout(0.05)
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()

        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0)

        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0)

        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0)

        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.LazyLinear(100)

        self.fc2 = nn.Linear(100, 50)

        self.fc3 = nn.Linear(50, 10)

        #steer throttle brake
        self.fc4 = nn.Linear(10, 3)


    def forward(self, x, command, speed):

        x = self.conv1(x)
        x = self.elu(x)

        x = self.conv2(x)
        x = self.elu(x)

        x = self.conv3(x)
        x = self.elu(x)

        x = self.conv4(x)
        x = self.elu(x)

        x = self.conv5(x)
        x = self.elu(x)

        x = self.flat(x)

        command = command.view(-1, 1).float()
        speed = speed.view(-1, 1).float()
        combined = torch.cat((x, command, speed), dim = 1)

        x = self.fc1(combined)
        x = self.dropout(x)
        x = self.elu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.elu(x)

        x = self.fc3(x)
        x = self.dropout(x)
        x = self.elu(x)

        x = self.fc4(x)

        return x

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        image, command, speed, labels = train_batch

        outputs = self.forward(image.float(), command.float(), speed.float())
        loss = self.loss_function(outputs, labels)

        self.log('train_loss', loss, on_step= True, on_epoch = True)

        mae_value = self.train_mae(outputs, labels)
        self.log('train_mae', mae_value, on_epoch=True, on_step= False)

        return loss

    def validation_step(self, train_batch, batch_idx):
        image, command, speed, labels = train_batch

        outputs = self.forward(image.float(), command.float(), speed.float())
        loss = self.loss_function(outputs, labels)

        self.log('val_loss', loss, on_step= True, on_epoch = True)

        mae_value = self.val_mae(outputs, labels)
        self.log('val_mae', mae_value, on_epoch=True, on_step= False)

        return loss

if __name__ == "__main__":
    location = Path(__file__).resolve().parent
    data_path = Path('../../labels/dave2')
    csv_path = data_path / 'final_annotations.csv'

    ddm = DrivingDataModule(str(csv_path))
    ddm.setup()

    # FOR DEBUG
    # count = [0, 0, 0 ,0, 0, 0]
    # for el in ddm.val_dataset:
    #     count[el[1]] += 1
    # print(len(ddm.val_dataset))
    # print(count)
    #
    # print(ddm.train_dataset[0][2])
    # import matplotlib.pyplot as plt
    # image = ddm.train_dataset[0][0]
    # plt.imshow(image.permute(1, 2, 0))
    # plt.show()


    driving_model = DrivingModel()

    from pytorch_lightning.loggers import TensorBoardLogger
    logger=TensorBoardLogger("../../logs", name="agent_dave2")
    trainer=pl.Trainer(logger=logger, max_epochs=10, log_every_n_steps=1, accelerator="gpu")
    trainer.fit(driving_model, ddm)