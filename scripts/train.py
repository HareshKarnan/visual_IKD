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
import torch.nn.functional as F

def croppatchinfront(image):
    return crop(image, 890, 584, 56, 100)

class L2Normalize(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1) # L2 normalize

from scripts.utils import \
    parse_bag_with_img, \
    filter_data, \
    process_joystick_data, \
    process_trackingcam_data, \
    process_accel_gyro_data


class VisualIKDNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(VisualIKDNet, self).__init__()
        self.flatten = nn.Flatten()

        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2), nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2), nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2), nn.PReLU(),
            self.flatten,
            nn.Linear(256*3*3, 128), nn.PReLU(),
            nn.Linear(128, 16), nn.Tanh()
        )

        self.trunk = nn.Sequential(
            nn.Linear(input_size + 16, hidden_size), nn.PReLU(),
            nn.Linear(hidden_size, hidden_size), nn.PReLU(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, non_image, image):
        print('image shape : ',image.shape)
        visual_embedding = self.visual_encoder(image)
        output = self.trunk(torch.cat((non_image, visual_embedding), dim=1))
        return output

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
    def __init__(self, data, batch_size):
        super(IKDDataModule, self).__init__()

        self.data = data
        print('length of data : ', len(self.data))
        self.batch_size = batch_size

        class MyDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data['odom'])-1

            def __getitem__(self, idx):
                odom = self.data['odom'][idx]
                joystick = self.data['joystick'][idx]
                accel = self.data['accel'][idx]
                gyro = self.data['gyro'][idx]
                bevimage = self.data['image'][idx]


                bevimage = cv2.resize(bevimage, (128, 128), interpolation=cv2.INTER_AREA)
                cv2.imshow('disp', bevimage)
                cv2.waitKey(1)

                return odom, joystick, accel, gyro, bevimage

        self.dataset = MyDataset(self.data)

        self.validation_dataset, self.training_dataset = random_split(self.dataset, [int(0.2*len(self.dataset)), len(self.dataset) - int(0.2*len(self.dataset))])

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True,
                          drop_last=not (len(self.training_dataset) % self.batch_size == 0.0))

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False,
                          drop_last=not (len(self.validation_dataset) % self.batch_size == 0.0))

# class IKDDataModule(pl.LightningDataModule):
#     def __init__(self, rosbag_path, frequency, max_time, config_path,
#                  topics_to_read, keys, batch_size, history_len):
#         super(IKDDataModule, self).__init__()
#         self.batch_size = batch_size
#         self.history_len = history_len
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             GaussianBlur(p=0.1),
#             transforms.Lambda(croppatchinfront),
#             transforms.Resize((64, 64)),
#         ])
#
#         with open(config_path, 'r') as f:
#             cprint('Reading Config file.. ', 'yellow')
#             config = yaml.safe_load(f)
#             cprint('Parsed Config file successfully ', 'yellow', attrs=['blink'])
#             print(config)
#
#         # parse all data from the rosbags
#         data, total_time = parse_bag_with_img(rosbag_path, topics_to_read, max_time=max_time)
#         # set the time intervals
#         times = np.linspace(0, max_time, frequency * max_time + 1)
#         # filter data based on time intervals
#         filtered_data = filter_data(data, times, keys, viz_images=False)
#         print('# filtered data points : ', len(filtered_data['rgb']))
#         # process joystick data
#         filtered_data = process_joystick_data(filtered_data, config=config)
#         # process tracking cam odom data
#         filtered_data = process_trackingcam_data(filtered_data)
#         # process accel and gyro data
#         filtered_data = process_accel_gyro_data(filtered_data)
#
#         # normalize odom and joystick
#         cprint('Normalizing joystick and odom values', 'red')
#         odom_mean = np.mean(filtered_data['odom'][:, :3], axis=0)
#         odom_std = np.std(filtered_data['odom'][:, :3], axis=0)
#
#         joystick_mean = np.mean(filtered_data['joystick'], axis=0)
#         joystick_std = np.std(filtered_data['joystick'], axis=0)
#
#         filtered_data['odom'][:, :3] = (filtered_data['odom'][:, :3] - odom_mean)/(odom_std + 1e-8)
#         filtered_data['joystick'] = (filtered_data['joystick'] - joystick_mean)/(joystick_std + 1e-8)
#
#         class MyDataset(Dataset):
#             def __init__(self, filtered_data, history_len):
#                 self.data = filtered_data
#                 self.history_len = history_len
#                 self.C_i = np.array(
#                     [622.0649233612024, 0.0, 633.1717569157071, 0.0, 619.7990184421728, 368.0688607187958, 0.0, 0.0,
#                      1.0]).reshape(
#                     (3, 3))
#
#             def __len__(self):
#                 return len(self.data['odom']) - self.history_len
#
#             def __getitem__(self, idx):
#                 odom_history = np.array([])
#                 for i in range(self.history_len):
#                     odom_history = np.concatenate((odom_history, self.data['odom'][idx+i][:3]))
#
#                 # joystick_history = np.array([])
#                 # for i in range(self.history_len):
#                 #     joystick_history = np.concatenate((joystick_history, self.data['joystick'][idx+i]))
#
#                 #############################################
#                 ### bird's eye view homography projection ###
#                 #############################################
#
#                 R_imu_world = R.from_quat(self.data['odom'][idx][-4:])
#                 R_imu_world = R_imu_world.as_euler('xyz', degrees=True)
#                 # R_imu_world[0] = 0.5
#                 # R_imu_world[1] = 0.
#                 R_imu_world[0], R_imu_world[1] = R_imu_world[0], -R_imu_world[1]
#                 R_imu_world[2] = 0.
#
#                 R_imu_world = R_imu_world
#                 R_imu_world = R.from_euler('xyz', R_imu_world, degrees=True)
#
#                 R_cam_imu = R.from_euler("xyz", [-90, 90, 0], degrees=True)
#                 R1 = R_cam_imu * R_imu_world
#                 R1 = R1.as_matrix()
#
#                 R2 = R.from_euler("xyz", [0, 0, -90], degrees=True).as_matrix()
#                 t1 = R1 @ np.array([0., 0., 0.5]).reshape((3, 1))
#                 t2 = R2 @ np.array([-2.5, -0., 6.0]).reshape((3, 1))
#                 n = np.array([0, 0, 1]).reshape((3, 1))
#                 n1 = R1 @ n
#
#                 H12 = self.homography_camera_displacement(R1, R2, t1, t2, n1)
#                 homography_matrix = self.C_i @ H12 @ np.linalg.inv(self.C_i)
#                 homography_matrix /= homography_matrix[2, 2]
#
#                 bev_img = cv2.warpPerspective(self.data['rgb'][idx],
#                                              homography_matrix, (1280, 720))
#
#                 # extract the image patch infront of the car
#                 bev_img = bev_img[420:520, 540:740]
#                 bev_img = cv2.resize(bev_img, (128, 128), interpolation=cv2.INTER_AREA)
#                 # bev_img = cv2.rectangle(bev_img, (540, 420), (740, 520), (0, 0, 255), thickness=2)
#
#                 # cv2.imshow('disp', bev_img)
#                 # cv2.waitKey(0)
#
#                 return odom_history, \
#                        self.data['joystick'][idx], \
#                        self.data['accel'][idx], \
#                        self.data['gyro'][idx], \
#                        bev_img/255.
#
#             @staticmethod
#             def homography_camera_displacement(R1, R2, t1, t2, n1):
#                 R12 = R2 @ R1.T
#                 t12 = R2 @ (- R1.T @ t1) + t2
#                 # d is distance from plane to t1.
#                 d = np.linalg.norm(n1.dot(t1.T))
#
#                 H12 = R12 - ((t12 @ n1.T) / d)
#                 H12 /= H12[2, 2]
#                 return H12
#
#         full_dataset = MyDataset(filtered_data, self.history_len)
#         dataset_len = len(full_dataset)
#         print('dataset length : ', dataset_len)
#
#         self.validation_dataset, self.train_dataset = random_split(full_dataset, [int(0.2 * dataset_len), dataset_len - int(0.2 * dataset_len)])
#
#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
#                           drop_last=not (len(self.train_dataset) % self.batch_size == 0.0))
#
#     def val_dataloader(self):
#         return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False,
#                           drop_last=not (len(self.validation_dataset) % self.batch_size == 0.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rosbag parser')
    parser.add_argument('--rosbag_path', type=str, default='data/ahgroad_new.bag')
    parser.add_argument('--frequency', type=int, default=20)
    parser.add_argument('--max_time', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--history_len', type=int, default=10)
    parser.add_argument('--config_path', type=str, default="config/alphatruck.yaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # accel + gyro + odom*history
    model = IKDModel(input_size=6 + 3*args.history_len,
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

    dm = IKDDataModule(data=data, batch_size=args.batch_size)

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


