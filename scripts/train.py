import os
import pickle

import numpy as np
import pytorch_lightning as pl
from termcolor import cprint
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datetime import datetime
import cv2
from scripts.model import VisualIKDNet, SimpleIKDNet
from scripts.arguments import get_args

class IKDModel(pl.LightningModule):
    def __init__(self, input_size, output_size,
                 hidden_size=64, use_vision=False):
        super(IKDModel, self).__init__()
        self.use_vision = use_vision
        cprint('input size :: '+str(input_size), 'green', attrs=['bold'])

        if use_vision:
            cprint('Using vision', 'green', attrs=['bold'])
            self.ikd_model = VisualIKDNet(input_size, output_size, hidden_size)
        else:
            cprint('Not using vision', 'green', attrs=['bold'])
            self.ikd_model = SimpleIKDNet(input_size, output_size, hidden_size)

        self.save_hyperparameters('input_size',
                                  'output_size',
                                  'hidden_size',
                                  'use_vision')

        self.loss = torch.nn.MSELoss()

    def forward(self, accel, gyro, odom, bevimage=None):
        if self.use_vision:
            # return self.ikd_model(non_visual_input, bevimage)
            return self.ikd_model(accel, gyro, odom, bevimage)
        else:
            # return self.ikd_model(non_visual_input)
            return self.ikd_model(accel, gyro, odom)

    def training_step(self, batch, batch_idx):
        odom, joystick, accel, gyro, bevimage = batch
        if self.use_vision:
            bevimage = bevimage.permute(0, 3, 1, 2)
            prediction = self.forward(accel.float(), gyro.float(), odom.float(), bevimage.float())
        else:
            prediction = self.forward(accel.float(), gyro.float(), odom.float())

        loss = self.loss(prediction, joystick.float())
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        odom, joystick, accel, gyro, bevimage = batch
        if self.use_vision:
            bevimage = bevimage.permute(0, 3, 1, 2)
            prediction = self.forward(accel.float(), gyro.float(), odom.float(), bevimage.float())
        else:
            prediction = self.forward(accel.float(), gyro.float(), odom.float())

        loss = self.loss(prediction, joystick.float())
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.ikd_model.parameters(), lr=3e-4, weight_decay=1e-5)

class ProcessedBagDataset(Dataset):
    def __init__(self, data, history_len):
        self.data = data
        self.history_len = history_len

        self.data['odom'] = np.asarray(self.data['odom'])
        self.data['joystick'] = np.asarray(self.data['joystick'])
        self.data['accel'] = np.asarray(self.data['accel'])
        self.data['gyro'] = np.asarray(self.data['gyro'])

        # self.data['joystick'][:, 0] = self.data['joystick'][:, 0] - self.data['odom'][:, 0]
        # self.data['joystick'][:, 1] = self.data['joystick'][:, 1] - self.data['odom'][:, 2]

        # odom_mean = np.mean(self.data['odom'], axis=0)
        # odom_std = np.std(self.data['odom'], axis=0)
        # joy_mean = np.mean(self.data['joystick'], axis=0)
        # joy_std = np.std(self.data['joystick'], axis=0)

        # self.data['odom'] = (self.data['odom'] - odom_mean) / odom_std
        # self.data['joystick'] = (self.data['joystick'] - joy_mean) / joy_std

    def __len__(self):
        return max(self.data['odom'].shape[0], 0)

    def __getitem__(self, idx):
        # history of odoms + next state
        odom_curr = self.data['odom'][idx][:3]
        odom_next = self.data['odom'][idx][-3:]
        odom_val = np.hstack((odom_curr, odom_next)).flatten()

        accel = self.data['accel'][idx]
        gyro = self.data['gyro'][idx]

        joystick = self.data['joystick'][idx]

        patches = self.data['patches'][idx]
        patch = patches[np.random.randint(0, len(patches))] # pick a random patch

        patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        patch /= 255.0

        # cv2.imshow('disp', patch)
        # cv2.waitKey(0)

        return odom_val, joystick, accel, gyro, patch

class IKDDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, train_dataset_names, val_dataset_names, batch_size, history_len):
        super(IKDDataModule, self).__init__()

        self.data_dir = data_dir
        self.train_dataset_names = train_dataset_names
        self.val_dataset_names = val_dataset_names
        self.batch_size = batch_size
        self.history_len = history_len

        train_datasets = []
        for dataset_name in self.train_dataset_names:
            data_files = os.listdir(os.path.join(data_dir, dataset_name))
            for file in data_files:
                data = pickle.load(open(os.path.join(data_dir, dataset_name, file), 'rb'))
                dataset = ProcessedBagDataset(data, history_len)
                if len(dataset) > 0:
                    train_datasets.append(dataset)

        self.training_dataset = torch.utils.data.ConcatDataset(train_datasets)
        cprint('Num training datapoints : '+str(len(self.training_dataset)), 'green', attrs=['bold'])
        # self.validation_dataset, self.training_dataset = random_split(self.dataset, [int(0.2*len(self.dataset)), len(self.dataset) - int(0.2*len(self.dataset))])

        val_datasets = []
        for dataset_name in self.val_dataset_names:
            data_files = os.listdir(os.path.join(data_dir, dataset_name))
            for file in data_files:
                data = pickle.load(open(os.path.join(data_dir, dataset_name, file), 'rb'))
                dataset = ProcessedBagDataset(data, history_len)
                if len(dataset) > 0:
                    val_datasets.append(dataset)

        self.validation_dataset = torch.utils.data.ConcatDataset(val_datasets)
        cprint('Num validation datapoints : '+str(len(self.validation_dataset)), 'green', attrs=['bold'])

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16,
                          drop_last=not (len(self.training_dataset) % self.batch_size == 0.0))

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16,
                          drop_last=not (len(self.validation_dataset) % self.batch_size == 0.0))

if __name__ == '__main__':
    # get the arguments
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IKDModel(input_size=3*60 + 3*200 + 3*(args.history_len+1), # accel + gyro + odom*history
                     output_size=2,
                     hidden_size=args.hidden_size,
                     use_vision=args.use_vision).to(device)

    model = model.to(device)
    dm = IKDDataModule(args.data_dir, args.train_dataset_names, args.val_dataset_names, batch_size=args.batch_size, history_len=args.history_len)

    early_stopping_cb = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00, patience=100)

    model_checkpoint_cb = ModelCheckpoint(dirpath='models/',
                                          filename=datetime.now().strftime("%d-%m-%Y-%H-%M-%S"),
                                          monitor='val_loss', verbose=True)

    print("Training model...")
    trainer = pl.Trainer(gpus=[0] if args.num_gpus==1 else [0, 1, 2, 3],
                         max_epochs=args.max_epochs,
                         callbacks=[early_stopping_cb, model_checkpoint_cb],
                         log_every_n_steps=10,
                        #  distributed_backend='ddp',
                         stochastic_weight_avg=False,
                         logger=True,
                         )

    trainer.fit(model, dm)
    
    # val_loader = dm.val_dataloader()
    # model.eval()
    # for d in val_loader:
    #     _, _, _, _, patch = d
    #     patch = patch.to(device).float()
    #     vis_embedding = model.visual_encoder(patch)
    #     # TODO write these out using tensorboard projector
    #
    #
    # cprint("Training complete..", 'green', attrs=['bold'])

