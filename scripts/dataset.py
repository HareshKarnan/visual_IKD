from torch.utils.data import Dataset
import numpy as np
import cv2

class MyDataset(Dataset):
    def __init__(self, data, history_len):
        self.data = data
        self.history_len = history_len

        self.data['odom'] = np.asarray(self.data['odom'])

        # self.data['joystick'][:, 0] = self.data['joystick'][:, 0] - self.data['odom'][:, 0]
        # self.data['joystick'][:, 1] = self.data['joystick'][:, 1] - self.data['odom'][:, 2]

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
        bevimage = self.data['image'][idx + self.history_len - 1]
        bevimage = cv2.resize(bevimage, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        bevimage /= 255.0

        # cv2.imshow('disp', bevimage)
        # cv2.waitKey(0)

        return np.asarray(odom_history).flatten(), joystick, accel, gyro, bevimage
