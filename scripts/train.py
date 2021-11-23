import glob
import os.path
import pickle

import pytorch_lightning as pl
import rosbag
import torch.nn as nn
import torch
import argparse
from termcolor import cprint
import yaml
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datetime import datetime
from scripts.utils import GaussianBlur
import cv2
from scipy.spatial.transform import Rotation as R
from scripts.quaternion import *
from scripts.model import VisualIKDNet
from torch.utils.data import ConcatDataset
from scripts.dataset import MyDataset

class IKDModel(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_size=64, history_len=1):
        super(IKDModel, self).__init__()
        self.visual_ikd_model = VisualIKDNet(input_size, output_size, hidden_size)

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomApply(
            #     [transforms.ColorJitter(brightness=0.4, contrast=0.4,
            #                             saturation=0.2, hue=0.1)],
            #     p=0.8
            # ),
        ])

        self.save_hyperparameters('input_size',
                                  'output_size',
                                  'hidden_size',
                                  'history_len')

        self.loss = torch.nn.MSELoss()

    def forward(self, non_visual_input, bevimage):
        return self.visual_ikd_model(non_visual_input, bevimage)

    def training_step(self, batch, batch_idx):
        odom, joystick, accel, gyro, bevimage = batch
        bevimage = bevimage.permute(0, 3, 1, 2)

        # apply augmentation here to the terrain patch
        bevimage = self.transform(bevimage)

        non_visual_input = torch.cat((odom, accel, gyro), dim=1)

        prediction = self.forward(non_visual_input.float(), bevimage.float())
        loss = self.loss(prediction, joystick.float())
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        odom, joystick, accel, gyro, bevimage = batch
        bevimage = bevimage.permute(0, 3, 1, 2)

        non_visual_input = torch.cat((odom, accel, gyro), dim=1)

        prediction = self.forward(non_visual_input.float(), bevimage.float())
        loss = self.loss(prediction, joystick.float())
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.visual_ikd_model.parameters(), lr=3e-5, weight_decay=1e-5)

class IKDDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, history_len):
        super(IKDDataModule, self).__init__()
        if not os.path.exists(data_path):
            raise Exception("Data does not exist at : " + data_path)
        self.batch_size = batch_size
        self.history_len = history_len

        # train dataset
        pickle_files = glob.glob(os.path.join(data_path, "train*_data.pkl"))
        print('Found ', len(pickle_files), ' train dataset files')
        datasets = []
        for pickle_file in pickle_files:
            data = pickle.load(open(pickle_file, 'rb'))
            datasets.append(MyDataset(data, self.history_len))
        self.training_dataset = ConcatDataset(datasets)

        # validation dataset
        pickle_files = glob.glob(os.path.join(data_path, "test*_data.pkl"))
        print('Found ', len(pickle_files), ' validation dataset files')
        datasets = []
        for pickle_file in pickle_files:
            data = pickle.load(open(pickle_file, 'rb'))
            datasets.append(MyDataset(data, self.history_len))
        self.validation_dataset = ConcatDataset(datasets)

        print('Num training data points : ', len(self.training_dataset))
        print('Num validation data points : ', len(self.validation_dataset))

        # self.validation_dataset, self.training_dataset = random_split(self.dataset, [int(0.2*len(self.dataset)), len(self.dataset) - int(0.2*len(self.dataset))])

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True,
                          drop_last=not (len(self.training_dataset) % self.batch_size == 0.0))

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False,
                          drop_last=not (len(self.validation_dataset) % self.batch_size == 0.0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rosbag parser')
    parser.add_argument('--rosbag_path', type=str, default='data/ahgroad_new.bag')
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--history_len', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--data_path', type=str, default="/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/ahg_indoor_bags/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # accel + gyro + odom*history
    model = IKDModel(input_size=3 + 3 + 3*(args.history_len+1),
                     output_size=2,
                     hidden_size=args.hidden_size,
                     history_len=args.history_len)

    model = model.to(device)

    dm = IKDDataModule(data_path=args.data_path, batch_size=args.batch_size, history_len=args.history_len)

    early_stopping_cb = EarlyStopping(monitor='val_loss',
                                      mode='min',
                                      min_delta=0.00,
                                      patience=100)

    model_checkpoint_cb = ModelCheckpoint(dirpath='models/',
                                          filename=datetime.now().strftime("%d-%m-%Y-%H-%M-%S"),
                                          monitor='val_loss', mode='min')

    print("Training model...")
    trainer = pl.Trainer(gpus=[0],
                         max_epochs=args.max_epochs,
                         callbacks=[early_stopping_cb, model_checkpoint_cb],
                         log_every_n_steps=10,
                         distributed_backend='ddp',
                         stochastic_weight_avg=False,
                         logger=True,
                         )

    trainer.fit(model, dm)



    # trainer.save_checkpoint('models/' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))


