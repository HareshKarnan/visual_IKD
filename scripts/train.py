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
from torchvision.transforms.functional import crop
from scripts.utils import GaussianBlur
import cv2
from scipy.spatial.transform import Rotation as R
from scripts.quaternion import *
from scripts.model import VisualIKDNet
import torch.nn.functional as F

def croppatchinfront(image):
    return crop(image, 890, 584, 56, 100)

class L2Normalize(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1) # L2 normalize

class IKDModel(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_size=64, history_len=1):
        super(IKDModel, self).__init__()
        self.visual_ikd_model = VisualIKDNet(input_size, output_size, hidden_size)

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
        return torch.optim.AdamW(self.visual_ikd_model.parameters(), lr=1e-5, weight_decay=1e-5)

class IKDDataModule(pl.LightningDataModule):
    def __init__(self, data, batch_size, history_len):
        super(IKDDataModule, self).__init__()

        self.data = data
        self.batch_size = batch_size
        self.history_len = history_len

        class MyDataset(Dataset):
            def __init__(self, data, history_len):
                self.data = data
                self.history_len = history_len

            def __len__(self):
                return len(self.data['odom']) - self.history_len

            def __getitem__(self, idx):
                # history of odoms + next state
                odom_history = self.data['odom'][idx:idx+self.history_len + 1]
                joystick = self.data['joystick'][idx + self.history_len - 1]
                accel = self.data['accel'][idx + self.history_len - 1]
                gyro = self.data['gyro'][idx + self.history_len - 1]
                bevimage = self.data['image'][idx + self.history_len - 1]
                bevimage = cv2.resize(bevimage, (128, 128), interpolation=cv2.INTER_AREA)

                # cv2.imshow('disp', bevimage)
                # cv2.waitKey(0)

                return np.asarray(odom_history).flatten(), joystick, accel, gyro, bevimage

        self.dataset = MyDataset(self.data, history_len)
        print('Total data points : ', len(self.dataset))

        self.validation_dataset, self.training_dataset = random_split(self.dataset, [int(0.2*len(self.dataset)), len(self.dataset) - int(0.2*len(self.dataset))])

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
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # accel + gyro + odom*history
    model = IKDModel(input_size=3 + 3 + 3*(args.history_len+1),
                     output_size=2,
                     hidden_size=256,
                     history_len=args.history_len)

    model = model.to(device)

    topics_to_read = [
        '/camera/odom/sample',
        '/joystick',
        '/camera/accel/sample',
        '/camera/gyro/sample',
        '/webcam/image_raw/compressed'
    ]

    keys = ['rgb', 'odom', 'accel', 'gyro', 'joystick']

    data = pickle.load(open('data/mydata.pkl', 'rb'))

    dm = IKDDataModule(data=data, batch_size=args.batch_size, history_len=args.history_len)

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
                         logger=True,
                         )

    trainer.fit(model, dm)



    # trainer.save_checkpoint('models/' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))


