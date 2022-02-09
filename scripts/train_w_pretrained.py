import torch
torch.backends.cudnn.benchmark = True
import os
from model import SimpleIKDNet
from arguments import get_args
import pickle
import numpy as np
import pytorch_lightning as pl
from termcolor import cprint
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datetime import datetime
import cv2
import argparse
import torch.nn as nn
from scripts.train_experience_encoder import EncoderModel
from collections import OrderedDict

class VisualIKDNetPretrained(nn.Module):
	def __init__(self, hidden_size, encoder):
		super(VisualIKDNetPretrained, self).__init__()
		self.encoder = encoder

		self.ikdmodel = nn.Sequential(
			nn.Linear(32 + 2 + 3, hidden_size), nn.BatchNorm1d(hidden_size), nn.PReLU(),
			nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.PReLU(),
			nn.Linear(hidden_size, hidden_size), nn.PReLU(),
			nn.Linear(hidden_size, 2),
		)

	def forward(self, accel, gyro, odom, visual_input, patch_observed):
		with torch.no_grad():
			visual_encoding = self.encoder.visual_imu_encoder_model.visual_encoder(visual_input)
			unobserved_indices = torch.nonzero(torch.logical_not(patch_observed)).squeeze()
			visual_encoding[unobserved_indices] = torch.zeros(visual_encoding[unobserved_indices].shape).cuda()
			imu_encoding = self.encoder.visual_imu_encoder_model.imu_net(torch.cat((accel, gyro), dim=1))
			kinodynamic_encoding = self.encoder.visual_imu_encoder_model.imu_visual_net(torch.cat((visual_encoding, imu_encoding), dim=1))

		return self.ikdmodel(torch.cat((kinodynamic_encoding, odom), dim=1))


class IKDModel(pl.LightningModule):
	def __init__(self, hidden_size=64, encoder_checkpoint_path=None):
		super(IKDModel, self).__init__()

		cprint('Loading the pretrained weights of the encoder', 'blue', attrs=['bold', 'blink'])
		self.encoder = EncoderModel.load_from_checkpoint(checkpoint_path=encoder_checkpoint_path)
		self.encoder.eval()
		# self.encoder = EncoderModel()
		self.ikd_model = VisualIKDNetPretrained(hidden_size, self.encoder)

		self.loss = torch.nn.MSELoss()

	def forward(self, accel, gyro, odom, bevimage=None, patch_observed=None):
		return self.ikd_model(accel, gyro, odom, bevimage, patch_observed)


	def training_step(self, batch, batch_idx):
		odom, joystick, accel, gyro, bevimage, patches_found = batch
		if bevimage is not None:
			bevimage = bevimage.permute(0, 3, 1, 2).float()
		prediction = self.forward(accel.float(), gyro.float(), odom.float(), bevimage, patches_found)

		loss = self.loss(prediction, joystick.float())
		self.log('train_loss', loss, prog_bar=True, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		odom, joystick, accel, gyro, bevimage, patches_found = batch
		if bevimage is not None:
			bevimage = bevimage.permute(0, 3, 1, 2).float()
		prediction = self.forward(accel.float(), gyro.float(), odom.float(), bevimage, patches_found)

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
		# self.data['odom_1sec_msg'] = np.asarray(self.data['odom_1sec_msg'])
		self.data['accel_msg'] = np.asarray(self.data['accel_msg'])
		self.data['gyro_msg'] = np.asarray(self.data['gyro_msg'])

	# self.data['joystick'][:, 0] = self.data['joystick'][:, 0] - self.data['odom'][:, 0]
	# self.data['joystick'][:, 1] = self.data['joystick'][:, 1] - self.data['odom'][:, 2]

	# odom_mean = np.mean(self.data['odom'], axis=0)
	# odom_std = np.std(self.data['odom'], axis=0)
	# joy_mean = np.mean(self.data['joystick'], axis=0)
	# joy_std = np.std(self.data['joystick'], axis=0)

	# self.data['odom'] = (self.data['odom'] - odom_mean) / odom_std
	# self.data['joystick'] = (self.data['joystick'] - joy_mean) / joy_std

	def __len__(self):
		return max(self.data['odom'].shape[0] - 6, 0)

	def __getitem__(self, idx):
		# history of odoms + next state
		odom_curr = self.data['odom'][idx][:3]
		odom_next = self.data['odom'][idx + 5][:3]

		odom_val = np.hstack((odom_curr,
							  odom_next[0],
							  odom_next[2])).flatten()

		# odom_1sec_history = self.data['odom_1sec_msg'][idx]
		accel = self.data['accel_msg'][idx]
		gyro = self.data['gyro_msg'][idx]
		joystick = self.data['joystick'][idx]
		patches = self.data['patches'][idx]
		patches_found = self.data['patches_found'][idx]

		patch = patches[np.random.randint(0, len(patches))]  # pick a random patch
		patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
		patch /= 255.0

		# cv2.imshow('disp', patch)
		# cv2.waitKey(0)

		# return odom_val, joystick-odom_val[-2:], accel, gyro, patch, patches_found
		return odom_val, joystick, accel, gyro, patch, patches_found


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
			data_files = [file for file in data_files if file.endswith('data_1.pkl')]
			for file in data_files:
				data = pickle.load(open(os.path.join(data_dir, dataset_name, file), 'rb'))
				dataset = ProcessedBagDataset(data, history_len)
				if len(dataset) > 0:
					train_datasets.append(dataset)

		self.training_dataset = torch.utils.data.ConcatDataset(train_datasets)
		cprint('Num training datapoints : ' + str(len(self.training_dataset)), 'green', attrs=['bold'])
		# self.validation_dataset, self.training_dataset = random_split(self.dataset, [int(0.2*len(self.dataset)), len(self.dataset) - int(0.2*len(self.dataset))])

		val_datasets = []
		for dataset_name in self.val_dataset_names:
			data_files = os.listdir(os.path.join(data_dir, dataset_name))
			data_files = [file for file in data_files if file.endswith('data_1.pkl')]
			for file in data_files:
				data = pickle.load(open(os.path.join(data_dir, dataset_name, file), 'rb'))
				dataset = ProcessedBagDataset(data, history_len)
				if len(dataset) > 0:
					val_datasets.append(dataset)

		self.validation_dataset = torch.utils.data.ConcatDataset(val_datasets)
		cprint('Num validation datapoints : ' + str(len(self.validation_dataset)), 'green', attrs=['bold'])

	def train_dataloader(self):
		return DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16,
						  drop_last=not (len(self.training_dataset) % self.batch_size == 0.0))

	def val_dataloader(self):
		return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16,
						  drop_last=not (len(self.validation_dataset) % self.batch_size == 0.0))


if __name__ == '__main__':
	# get the arguments
	parser = argparse.ArgumentParser(description='rosbag parser')
	parser.add_argument('--max_epochs', type=int, default=1000)
	parser.add_argument('--history_len', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--num_gpus', type=int, default=1)
	parser.add_argument('--hidden_size', type=int, default=32)
	parser.add_argument('--use_vision', action='store_true', default=True)
	parser.add_argument('--data_dir', type=str, default='/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/ahg_indoor_bags/')
	parser.add_argument('--train_dataset_names', type=str, nargs='+', default=['train1_data',
																			   'train2_data',
																			   'train3_data',
																			   'train5_data',
																			   'train7_data',
																			   'train9_data',
																			   'train10_data'])
	parser.add_argument('--val_dataset_names', type=str, nargs='+', default=['train4_data',
																			 'train6_data',
																			 'train8_data'])
	parser.add_argument('--encoder_checkpoint', type=str, default='models/encoder/31-01-2022-21-17-37.ckpt')
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = IKDModel(hidden_size=args.hidden_size, encoder_checkpoint_path=args.encoder_checkpoint).cuda()

	model = model.cuda()
	dm = IKDDataModule(args.data_dir, args.train_dataset_names, args.val_dataset_names, batch_size=args.batch_size,
					   history_len=args.history_len)

	early_stopping_cb = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00, patience=100)

	model_checkpoint_cb = ModelCheckpoint(dirpath='models/ikd_w_pretrained/',
										  filename=datetime.now().strftime("%d-%m-%Y-%H-%M-%S"),
										  monitor='val_loss', verbose=True)

	print("Training model...")
	trainer = pl.Trainer(gpus=list(np.arange(args.num_gpus)),
						 max_epochs=args.max_epochs,
						 callbacks=[early_stopping_cb, model_checkpoint_cb],
						 log_every_n_steps=10,
						 distributed_backend='ddp',
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
