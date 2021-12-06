import pickle

import matplotlib.pyplot as plt
import pytorch_lightning as pl
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
import cv2
from scipy.spatial.transform import Rotation as R
from scripts.quaternion import *
from scripts.model import VisualIKDNet
import torch.nn.functional as F
from tqdm import tqdm
from scripts.train import IKDModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rosbag parser')
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--history_len', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IKDModel.load_from_checkpoint('models/06-12-2021-11-49-19.ckpt')

    model = model.to(device)

    data = pickle.load(open('/robodata/kvsikand/visualIKD/train1_data/data_1.pkl', 'rb'))

    class ProcessedBagDataset(Dataset):
        def __init__(self, data, history_len):
            self.data = data
            self.history_len = history_len


            self.data['odom'] = np.asarray(self.data['odom'])
            self.data['joystick'] = np.asarray(self.data['joystick'])
            # odom_mean = np.mean(self.data['odom'], axis=0)
            # odom_std = np.std(self.data['odom'], axis=0)
            # joy_mean = np.mean(self.data['joystick'], axis=0)
            # joy_std = np.std(self.data['joystick'], axis=0)

            # self.data['odom'] = (self.data['odom'] - odom_mean) / odom_std
            # self.data['joystick'] = (self.data['joystick'] - joy_mean) / joy_std

        def __len__(self):
            return len(self.data['odom']) - self.history_len

        def __getitem__(self, idx):
            # history of odoms + next state
            odom_history = self.data['odom'][idx:idx + self.history_len + 1]
            joystick = self.data['joystick'][idx + self.history_len - 1]
            accel = self.data['accel'][idx + self.history_len - 1]
            gyro = self.data['gyro'][idx + self.history_len - 1]
            patch = self.data['patches'][idx + self.history_len - 1]
            patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)

            patch = patch.astype(np.float32) / 255.0

            return np.asarray(odom_history).flatten(), joystick, accel, gyro, patch

    dataset = ProcessedBagDataset(data, args.history_len)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            drop_last=not (len(dataset) % args.batch_size == 0.0))

    joystick_true, joystick_pred = [], []
    for odom_history, joystick, accel, gyro, bevimage in tqdm(dataloader):
        with torch.no_grad():
            bevimage = bevimage.permute(0, 3, 1, 2).to(device)
            non_visual_input = torch.cat((odom_history, accel, gyro), dim=1).to(device)
            output = model.forward(non_visual_input.float(), bevimage.float())
        output = output.cpu().numpy()

        joystick_true.append(joystick.numpy().flatten())
        joystick_pred.append(np.asarray(output[0]))

    joystick_true = np.asarray(joystick_true)
    joystick_pred = np.asarray(joystick_pred)

    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(joystick_true)), joystick_true[:, 0], label='true', color='blue')
    plt.plot(np.arange(len(joystick_true)), joystick_pred[:, 0], label='pred', color='red')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(joystick_true)), joystick_true[:, 1], label='true', color='blue')
    plt.plot(np.arange(len(joystick_true)), joystick_pred[:, 1], label='pred', color='red')
    plt.savefig('graph.png')
