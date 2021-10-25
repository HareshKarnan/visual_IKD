import pytorch_lightning as pl
import torch.nn as nn
import torch
import argparse
from termcolor import cprint
import yaml
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datetime import datetime

from scripts.utils import \
    parse_bag_with_img, \
    filter_data, \
    process_joystick_data, \
    process_trackingcam_data, \
    process_accel_gyro_data

class IKDModel(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(IKDModel, self).__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.PReLU(),
            # nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size), nn.PReLU(),
            nn.Linear(hidden_size, hidden_size), nn.PReLU(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

        self.save_hyperparameters('input_size',
                                  'output_size',
                                  'hidden_size')

        self.loss = torch.nn.SmoothL1Loss()

    def training_step(self, batch, batch_idx):
        odom, joystick, accel, gyro, _ = batch
        input = torch.cat((odom, accel, gyro), dim=1)

        prediction = self.trunk(input.float())
        loss = self.loss(prediction, joystick.float())
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        odom, joystick, accel, gyro, _ = batch

        input = torch.cat((odom, accel, gyro), dim=1)

        prediction = self.trunk(input.float())
        loss = self.loss(prediction, joystick.float())
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the test_step for validating
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=3e-4)
        return optimizer

class IKDDataModule(pl.LightningDataModule):
    def __init__(self, rosbag_path, frequency, max_time, config_path, topics_to_read, keys, batch_size):
        super(IKDDataModule, self).__init__()
        self.batch_size = batch_size

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
        # process tracking cam data
        filtered_data = process_trackingcam_data(filtered_data)
        # process accel and gyro data
        filtered_data = process_accel_gyro_data(filtered_data)

        # normalize odom and joystick
        cprint('Normalizing joystick and odom values', 'red')
        odom_mean = np.mean(filtered_data['odom'], axis=0)
        odom_std = np.std(filtered_data['odom'], axis=0)

        joystick_mean = np.mean(filtered_data['joystick'], axis=0)
        joystick_std = np.std(filtered_data['joystick'], axis=0)

        filtered_data['odom'] = (filtered_data['odom'] - odom_mean)/(odom_std + 1e-8)
        filtered_data['joystick'] = (filtered_data['joystick'] - joystick_mean)/(joystick_std + 1e-8)

        class MyDataset(Dataset):
            def __init__(self, filtered_data):
                self.data = filtered_data

            def __len__(self):
                return len(self.data['odom'])

            def __getitem__(self, idx):
                return self.data['odom'][idx], self.data['joystick'][idx], self.data['accel'][idx], \
                       self.data['gyro'][idx], self.data['rgb'][idx]

        full_dataset = MyDataset(filtered_data)
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
    parser.add_argument('--rosbag_path', type=str, default='data/ahg_road.bag')
    parser.add_argument('--frequency', type=int, default=20)
    parser.add_argument('--max_time', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--config_path', type=str, default="config/alphatruck.yaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IKDModel(input_size=6+3, output_size=2, hidden_size=256)
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
                       topics_to_read = topics_to_read,
                       keys=keys, batch_size=args.batch_size)

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


