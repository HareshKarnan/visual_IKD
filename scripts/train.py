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
        visual_embedding = self.visual_encoder(image)
        output = self.trunk(torch.cat((non_image, visual_embedding), dim=1))
        return output

class IKDModel(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(IKDModel, self).__init__()
        self.visual_ikd_model = VisualIKDNet(input_size, output_size, hidden_size)

        self.save_hyperparameters('input_size',
                                  'output_size',
                                  'hidden_size')

        self.loss = torch.nn.SmoothL1Loss()

        self.K = np.array(
            [622.0649233612024, 0.0, 633.1717569157071, 0.0, 619.7990184421728, 368.0688607187958, 0.0, 0.0,
             1.0]).reshape(
            (3, 3))
        self.C_i = torch.from_numpy(self.K).float().to(self.device)

    def forward(self, non_visual_input, bevimage):
        return self.visual_ikd_model(non_visual_input, bevimage)

    # def img_fpv_to_bev(self, image, odom):
    #     """
    #     function to convert a first person view image into a bev image
    #     :param image: input first person view image
    #     :param odom: input rotation from the intel realsense (w,x,y,z)
    #     :return: bird's eye view image
    #     """
    #
    #     R_imu_world = qeuler(odom, 'xyz')
    #     R_imu_world[:, 0], R_imu_world[:, 1] = R_imu_world[:, 0], -R_imu_world[:, 1]
    #     R_imu_world[:, 2] = 0.
    #
    #     R_imu_world = euler_to_quaternion(R_imu_world, 'xyz')
    #
    #     R_cam_imu = torch.tensor([[0.5, -0.5, 0.5, -0.5]], dtype=torch.float32, device=self.device)
    #     R_cam_imu.repeat(R_imu_world.shape[0], 1)
    #
    #     R1 = qmul(R_cam_imu, R_imu_world)
    #     R1 = quaternion_to_matrix(R1)
    #
    #     R2 = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=torch.float32, device=self.device)
    #     R2 = R2.repeat(R_imu_world.shape[0], 1, 1)
    #
    #     t1 = R1 @ torch.tensor([[0, 0, 0.5]], dtype=torch.float32, device=self.device)
    #     t2 = R2 @ torch.tensor([[-2.5, 0, 6.0]], dtype=torch.float32, device=self.device)
    #     n = torch.tensor([[0, 0, 1]], dtype=torch.float32, device=self.device).reshape(3, 1)
    #     n1 = R1 @ n
    #
    #     R12 = R2 @ R1.T
    #     t12 = R2 @ (-R1.T @ t1) + t2
    #     # d is distance from plane to t1
    #     d = torch.norm(n1.dot(t1.T))
    #
    #     H12 = R12 - ((t12 @ n1.T) / d)
    #     H12 /= H12[2, 2]
    #
    #     homography_matrix = self.C_i @ H12 @ torch.inverse(self.C_i)
    #     homography_matrix /= homography_matrix[2, 2]
    #
    #     image = kornia.warp_perspective(image, homography_matrix, dsize=(1280, 720))
    #
    #     return image

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
        # Here we just reuse the test_step for validating
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=3e-4)
        return optimizer

class IKDDataModule(pl.LightningDataModule):
    def __init__(self, rosbag_path, frequency, max_time, config_path,
                 topics_to_read, keys, batch_size, history_len):
        super(IKDDataModule, self).__init__()
        self.batch_size = batch_size
        self.history_len = history_len
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            GaussianBlur(p=0.1),
            transforms.Lambda(croppatchinfront),
            transforms.Resize((64, 64)),
        ])

        with open(config_path, 'r') as f:
            cprint('Reading Config file.. ', 'yellow')
            config = yaml.safe_load(f)
            cprint('Parsed Config file successfully ', 'yellow', attrs=['blink'])
            print(config)

        # parse all data from the rosbags
        data, total_time = parse_bag_with_img(rosbag_path, topics_to_read, max_time=max_time)
        # set the time intervals
        times = np.linspace(0, max_time, frequency * max_time + 1)
        # filter data based on time intervals
        filtered_data = filter_data(data, times, keys, viz_images=False)
        print('# filtered data points : ', len(filtered_data['rgb']))
        # process joystick data
        filtered_data = process_joystick_data(filtered_data, config=config)
        # process tracking cam odom data
        filtered_data = process_trackingcam_data(filtered_data)
        # process accel and gyro data
        filtered_data = process_accel_gyro_data(filtered_data)

        # normalize odom and joystick
        cprint('Normalizing joystick and odom values', 'red')
        odom_mean = np.mean(filtered_data['odom'][:, :3], axis=0)
        odom_std = np.std(filtered_data['odom'][:, :3], axis=0)

        joystick_mean = np.mean(filtered_data['joystick'], axis=0)
        joystick_std = np.std(filtered_data['joystick'], axis=0)

        filtered_data['odom'][:, :3] = (filtered_data['odom'][:, :3] - odom_mean)/(odom_std + 1e-8)
        filtered_data['joystick'] = (filtered_data['joystick'] - joystick_mean)/(joystick_std + 1e-8)

        class MyDataset(Dataset):
            def __init__(self, filtered_data, history_len):
                self.data = filtered_data
                self.history_len = history_len
                self.C_i = np.array(
                    [622.0649233612024, 0.0, 633.1717569157071, 0.0, 619.7990184421728, 368.0688607187958, 0.0, 0.0,
                     1.0]).reshape(
                    (3, 3))

            def __len__(self):
                return len(self.data['odom']) - self.history_len

            def __getitem__(self, idx):
                odom_history = np.array([])
                for i in range(self.history_len):
                    odom_history = np.concatenate((odom_history, self.data['odom'][idx+i][:3]))

                joystick_history = np.array([])
                for i in range(self.history_len):
                    joystick_history = np.concatenate((joystick_history, self.data['joystick'][idx+i]))

                #############################################
                ### bird's eye view homography projection ###
                #############################################

                R_imu_world = R.from_quat(self.data['odom'][idx][-4:])
                R_imu_world = R_imu_world.as_euler('xyz', degrees=True)
                # R_imu_world[0] = 0.5
                # R_imu_world[1] = 0.
                R_imu_world[0], R_imu_world[1] = R_imu_world[0], -R_imu_world[1]
                R_imu_world[2] = 0.

                R_imu_world = R_imu_world
                R_imu_world = R.from_euler('xyz', R_imu_world, degrees=True)

                R_cam_imu = R.from_euler("xyz", [-90, 90, 0], degrees=True)
                R1 = R_cam_imu * R_imu_world
                R1 = R1.as_matrix()

                R2 = R.from_euler("xyz", [0, 0, -90], degrees=True).as_matrix()
                t1 = R1 @ np.array([0., 0., 0.5]).reshape((3, 1))
                t2 = R2 @ np.array([-2.5, -0., 6.0]).reshape((3, 1))
                n = np.array([0, 0, 1]).reshape((3, 1))
                n1 = R1 @ n

                H12 = self.homography_camera_displacement(R1, R2, t1, t2, n1)
                homography_matrix = self.C_i @ H12 @ np.linalg.inv(self.C_i)
                homography_matrix /= homography_matrix[2, 2]

                bev_img = cv2.warpPerspective(self.data['rgb'][idx],
                                             homography_matrix, (1280, 720))

                # extract the image patch infront of the car
                bev_img = bev_img[420:520, 540:740]
                bev_img = cv2.resize(bev_img, (128, 128), interpolation=cv2.INTER_AREA)
                # bev_img = cv2.rectangle(bev_img, (540, 420), (740, 520), (0, 0, 255), thickness=2)

                # cv2.imshow('disp', bev_img)
                # cv2.waitKey(0)

                return odom_history, \
                       joystick_history, \
                       self.data['accel'][idx], \
                       self.data['gyro'][idx], \
                       bev_img/255.

            @staticmethod
            def homography_camera_displacement(R1, R2, t1, t2, n1):
                R12 = R2 @ R1.T
                t12 = R2 @ (- R1.T @ t1) + t2
                # d is distance from plane to t1.
                d = np.linalg.norm(n1.dot(t1.T))

                H12 = R12 - ((t12 @ n1.T) / d)
                H12 /= H12[2, 2]
                return H12

        full_dataset = MyDataset(filtered_data, self.history_len)
        dataset_len = len(full_dataset)
        print('dataset length : ', dataset_len)

        self.validation_dataset, self.train_dataset = random_split(full_dataset, [int(0.2 * dataset_len), dataset_len - int(0.2 * dataset_len)])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          drop_last=not (len(self.train_dataset) % self.batch_size == 0.0))

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False,
                          drop_last=not (len(self.validation_dataset) % self.batch_size == 0.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rosbag parser')
    parser.add_argument('--rosbag_path', type=str, default='data/ahgroad_new.bag')
    parser.add_argument('--frequency', type=int, default=20)
    parser.add_argument('--max_time', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--history_len', type=int, default=20)
    parser.add_argument('--config_path', type=str, default="config/alphatruck.yaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # accel + gyro + odom*history
    model = IKDModel(input_size=6 + 3*args.history_len, output_size=2*args.history_len, hidden_size=256)
    model = model.to(device)

    topics_to_read = [
        '/camera/odom/sample',
        '/joystick',
        '/camera/accel/sample',
        '/camera/gyro/sample',
        '/webcam/image_raw/compressed'
    ]

    keys = ['rgb', 'odom', 'accel', 'gyro', 'joystick']

    dm = IKDDataModule(rosbag_path=args.rosbag_path,
                       frequency=args.frequency,
                       max_time=args.max_time,
                       config_path=args.config_path,
                       topics_to_read=topics_to_read,
                       keys=keys, batch_size=args.batch_size,
                       history_len=args.history_len)

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


