import pickle

import matplotlib.pyplot as plt
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
from tqdm import tqdm
from scripts.train import IKDModel
from scripts.dataset import MyDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rosbag parser')
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--history_len', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IKDModel.load_from_checkpoint('models/23-11-2021-12-18-53.ckpt')

    model = model.to(device)

    data = pickle.load(open('/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/ahg_indoor_bags/test1_data.pkl', 'rb'))

    dataset = MyDataset(data, args.history_len)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            drop_last=not (len(dataset) % args.batch_size == 0.0))

    joystick_true, joystick_pred = [], []
    for odom_history, joystick, accel, gyro, bevimage in tqdm(dataloader):
        with torch.no_grad():
            bevimage = bevimage.permute(0, 3, 1, 2).to(device)
            non_visual_input = torch.cat((odom_history, accel, gyro), dim=1).to(device)
            # output = model.forward(non_visual_input.float(), bevimage.float())
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
    plt.show()
