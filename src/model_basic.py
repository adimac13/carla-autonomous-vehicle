import pandas as pd
import torch
import cv2
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
        transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        full_dataset = DrivingDataset(self.csv_path, transform)
        self.val_dataset, self.train_dataset = random_split(full_dataset,
                                                             [len(full_dataset)//10, len(full_dataset) - len(full_dataset)//10])
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

location = Path(__file__).resolve().parent
data_path = Path('../labels')
csv_path = data_path / 'final_annotations.csv'

ddm = DrivingDataModule(str(csv_path))
ddm.setup()

#FOR DEBUG
# count = [0, 0, 0 ,0, 0, 0]
# for el in ddm.val_dataset:
#     count[el[1]] += 1
# print(len(ddm.val_dataset))
# print(count)

# print(ddm.train_dataset[0][2])
# import matplotlib.pyplot as plt
# image = ddm.train_dataset[0][0]
# plt.imshow(image.permute(1, 2, 0))
# plt.show()

import torchmetrics
from torch import nn
from torch import optim
import torch.nn.functional as F

# class DrivingModule(pl.LightningModule):
#     def __init__(self):
#         super().__init__()





