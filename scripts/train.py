import os
import pickle

import pytorch_lightning as pl
from termcolor import cprint
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datetime import datetime
import cv2
from scripts.old.quaternion import *
from scripts.model import VisualIKDNet, SimpleIKDNet
from scripts.arguments import get_args

class IKDModel(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_size=64, use_vision=False):
        super(IKDModel, self).__init__()

        if use_vision:
            self.ikd_model = VisualIKDNet(input_size, output_size, hidden_size)
        else:
            self.ikd_model = SimpleIKDNet(input_size, output_size, hidden_size)

        self.save_hyperparameters('input_size',
                                  'output_size',
                                  'hidden_size')

        self.loss = torch.nn.MSELoss()

    def forward(self, non_visual_input, bevimage):
        return self.ikd_model(non_visual_input, bevimage)

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
        return torch.optim.AdamW(self.ikd_model.parameters(), lr=3e-4, weight_decay=1e-5)

class IKDDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, dataset_names, batch_size, history_len):
        super(IKDDataModule, self).__init__()

        self.data_dir = data_dir
        self.dataset_names = dataset_names
        self.batch_size = batch_size
        self.history_len = history_len

        class ProcessedBagDataset(Dataset):
            def __init__(self, data, history_len):
                self.data = data
                self.history_len = history_len

                self.data['odom'] = np.asarray(self.data['odom'])
                self.data['joystick'] = np.asarray(self.data['joystick'])                # self.data['joystick'][:, 0] = self.data['joystick'][:, 0] - self.data['odom'][:, 0]
                # self.data['joystick'][:, 1] = self.data['joystick'][:, 1] - self.data['odom'][:, 2]

                # odom_mean = np.mean(self.data['odom'], axis=0)
                # odom_std = np.std(self.data['odom'], axis=0)
                # joy_mean = np.mean(self.data['joystick'], axis=0)
                # joy_std = np.std(self.data['joystick'], axis=0)

                # self.data['odom'] = (self.data['odom'] - odom_mean) / odom_std
                # self.data['joystick'] = (self.data['joystick'] - joy_mean) / joy_std

            def __len__(self):
                return max(self.data['joystick'].shape[0] - self.history_len, 0)

            def __getitem__(self, idx):
                # history of odoms + next state
                odom_history = self.data['odom'][idx:idx+self.history_len + 1]
                joystick = self.data['joystick'][idx + self.history_len - 1]
                accel = self.data['accel'][idx + self.history_len - 1]
                gyro = self.data['gyro'][idx + self.history_len - 1]
                patches = self.data['patches'][idx + self.history_len - 1]
                rand_idx = np.random.randint(0, len(patches))
                patch = patches[rand_idx]
                patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
                patch /= 255.0

                # cv2.imshow('disp', patch)
                # cv2.waitKey(0)

                return np.asarray(odom_history).flatten(), joystick, accel, gyro, patch

        datasets = []
        for dataset_name in dataset_names:
            data_files = os.listdir(os.path.join(data_dir, dataset_name))
            for file in data_files:
                data = pickle.load(open(os.path.join(data_dir, dataset_name, file), 'rb'))
                dataset = ProcessedBagDataset(data, history_len)
                if len(dataset) > 0:
                    datasets.append(dataset)

        self.dataset = torch.utils.data.ConcatDataset(datasets)
        print('Total data points : ', len(self.dataset))

        self.validation_dataset, self.training_dataset = random_split(self.dataset, [int(0.2*len(self.dataset)), len(self.dataset) - int(0.2*len(self.dataset))])

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16,
                          drop_last=not (len(self.training_dataset) % self.batch_size == 0.0))

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16,
                          drop_last=not (len(self.validation_dataset) % self.batch_size == 0.0))

if __name__ == '__main__':
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description='rosbag parser')
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--history_len', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--use_vision', action='store_true', default=False)
    parser.add_argument('--data_dir', type=str, default='/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/ahg_indoor_bags/')
    parser.add_argument('--dataset_names', type=str, nargs='+', default=['train1'])
    args = parser.parse_args()
=======
    # get the arguments
    args = get_args()
>>>>>>> 0144202f6e19ae021345ffdeaab916466ef32ea6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IKDModel(input_size=3 + 3 + 3*(args.history_len+1), # accel + gyro + odom*history
                     output_size=2,
                     hidden_size=args.hidden_size,
                     use_vision=args.use_vision).to(device)

    model = model.to(device)
    dm = IKDDataModule(args.data_dir, args.dataset_names, batch_size=args.batch_size, history_len=args.history_len)

    early_stopping_cb = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00, patience=100)

    model_checkpoint_cb = ModelCheckpoint(dirpath='models/', filename=datetime.now().strftime("%d-%m-%Y-%H-%M-%S"),
                                          monitor='val_loss')

    print("Training model...")
    trainer = pl.Trainer(gpus=[0],
                         max_epochs=args.max_epochs,
                         callbacks=[early_stopping_cb, model_checkpoint_cb],
                         log_every_n_steps=10,
                        #  distributed_backend='ddp',
                         stochastic_weight_avg=False,
                         logger=True,
                         )

    trainer.fit(model, dm)

    cprint("Training complete..", 'green', attrs=['bold'])

