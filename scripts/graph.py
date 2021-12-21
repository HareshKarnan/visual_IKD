import pickle

import matplotlib.pyplot as plt
import argparse
from torch.utils.data import Dataset, DataLoader
import cv2
from scripts.old.quaternion import *
from tqdm import tqdm
from scripts.train import IKDModel, ProcessedBagDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rosbag parser')
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--history_len', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--use_vision', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IKDModel.load_from_checkpoint('models/20-12-2021-16-41-18.ckpt', use_vision=args.use_vision)
    model = model.to(device)
    model.eval()

    data = pickle.load(open('/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/ahg_indoor_bags/train3_data/data_1.pkl', 'rb'))

    # class ProcessedBagDataset(Dataset):
    #     def __init__(self, data, history_len):
    #         self.data = data
    #         self.history_len = history_len
    #
    #
    #         self.data['odom'] = np.asarray(self.data['odom'])
    #         self.data['joystick'] = np.asarray(self.data['joystick'])
    #         # odom_mean = np.mean(self.data['odom'], axis=0)
    #         # odom_std = np.std(self.data['odom'], axis=0)
    #         # joy_mean = np.mean(self.data['joystick'], axis=0)
    #         # joy_std = np.std(self.data['joystick'], axis=0)
    #
    #         # self.data['odom'] = (self.data['odom'] - odom_mean) / odom_std
    #         # self.data['joystick'] = (self.data['joystick'] - joy_mean) / joy_std
    #
    #     def __len__(self):
    #         return len(self.data['odom']) - self.history_len
    #
    #     def __getitem__(self, idx):
    #         # history of odoms + next state
    #         odom_history = self.data['odom'][idx:idx + self.history_len + 1]
    #         joystick = self.data['joystick'][idx + self.history_len - 1]
    #         accel = self.data['accel'][idx + self.history_len - 1]
    #         gyro = self.data['gyro'][idx + self.history_len - 1]
    #         patch = self.data['patches'][idx + self.history_len - 1]
    #         patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
    #
    #         patch = patch.astype(np.float32) / 255.0
    #
    #         return np.asarray(odom_history).flatten(), joystick, accel, gyro, patch

    dataset = ProcessedBagDataset(data, args.history_len)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            drop_last=not (len(dataset) % args.batch_size == 0.0))

    joystick_true, joystick_pred = [], []
    for odom_history, joystick, accel, gyro, bevimage in tqdm(dataloader):
        odom_history = odom_history.to(device).float()
        accel = accel.to(device).float()
        gyro = gyro.to(device).float()
        with torch.no_grad():
            # non_visual_input = torch.cat((odom_history, accel, gyro), dim=1).to(device)
            if args.use_vision:
                bevimage = bevimage.permute(0, 3, 1, 2).to(device)
                output = model.forward(accel, gyro, odom_history, bevimage.float())
            else:
                output = model.forward(accel, gyro, odom_history)
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
    # plt.savefig('graph.png')
    plt.show()