import copy

import torch
import numpy as np
import os
import argparse
torch.backends.cudnn.benchmark = True
import pytorch_lightning as pl
from termcolor import cprint
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pickle
import glob
import torch.nn as nn
from datetime import datetime
import random
import cv2
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import torch.nn.functional as F

print('CUDA AVAILABLE : ', torch.cuda.is_available())

class VisualIMUEncoder(nn.Module):
	def __init__(self):
		super(VisualIMUEncoder, self).__init__()
		self.visual_encoder = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, stride=2), nn.ReLU(),  # 31x31
			nn.MaxPool2d(kernel_size=3, stride=2),  # 15x15
			nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(),  # 7x7
			nn.MaxPool2d(kernel_size=3, stride=2),  # 3x3
			nn.Flatten(),
			nn.Linear(3 * 3 * 32, 128), nn.ReLU(),
			nn.Linear(128, 64)
		)

		self.imu_net = nn.Sequential(
			nn.Linear(200 * 3 + 60 * 3, 128), nn.ReLU(),
			nn.Linear(128, 64), nn.ReLU(),
			nn.Linear(64, 32),
		)

		self.imu_visual_net = nn.Sequential(
			nn.Linear(64 + 32 + 2*4, 64), nn.ReLU(), # 64 visual + 32 imu + 2*4 action history
			nn.Linear(64, 64), nn.ReLU(),
			nn.Linear(64, 6),
		)

	def forward(self, visual_input, imu_input, action_history):
		patch_embedding = self.visual_encoder(visual_input)
		imu_embedding = self.imu_net(imu_input)
		# print('patch shape : ', patch_embedding.shape)
		# print('imu shape : ', imu_embedding.shape)
		# print('action history shape : ', action_history.shape)
		embedding = self.imu_visual_net(torch.cat([patch_embedding, imu_embedding, action_history], dim=1))
		# embedding = F.normalize(embedding, p=2, dim=1)
		return embedding

class EncoderModel(pl.LightningModule):
	def __init__(self, args=None, margin=0.2, save_hyperparam=False):
		super(EncoderModel, self).__init__()
		self.save_hyperparameters('args')
		self.visual_imu_encoder_model = VisualIMUEncoder()
		self.loss = nn.TripletMarginLoss(margin=margin, swap=False, p=2)
		# self.loss = self.softtripletloss

	def softtripletloss(self, an_embedding, pos_embedding, neg_embedding):
		d_ap = torch.sum((an_embedding - pos_embedding)**2, dim=1)
		d_an = torch.sum((an_embedding - neg_embedding)**2, dim=1)
		soft_triplet_loss = (torch.exp(d_ap) / (torch.exp(d_ap) + torch.exp(d_an)))**2
		return soft_triplet_loss.mean()

	def forward(self, patch, imu, action):
		return self.visual_imu_encoder_model(patch, imu, action)

	def training_step(self, batch, batch_idx):
		anchor_patch, positive_patch, negative_patch, anchor_imu, positive_imu, negative_imu, _, anchor_action, positive_action, negative_action = batch

		an_embedding = self.forward(anchor_patch.float(), anchor_imu.float(), anchor_action.float())
		pos_embedding = self.forward(positive_patch.float(), positive_imu.float(), positive_action.float())
		neg_embedding = self.forward(negative_patch.float(), negative_imu.float(), negative_action.float())
		loss = self.loss(an_embedding, pos_embedding, neg_embedding)
		self.log('train_loss', loss, prog_bar=True, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		anchor_patch, positive_patch, negative_patch, anchor_imu, positive_imu, negative_imu, _, anchor_action, positive_action, negative_action = batch
		an_embedding = self.forward(anchor_patch.float(), anchor_imu.float(), anchor_action.float())
		pos_embedding = self.forward(positive_patch.float(), positive_imu.float(), positive_action.float())
		neg_embedding = self.forward(negative_patch.float(), negative_imu.float(), negative_action.float())
		loss = self.loss(an_embedding, pos_embedding, neg_embedding)
		self.log('val_loss', loss, prog_bar=True, logger=True)
		return loss

	@staticmethod
	def write_metadata_on_image(img, odom_curr, joystick, odom_next):
		img = cv2.putText(img, 'curr : ' + str(odom_curr), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
		img = cv2.putText(img, 'joy : ' + str(joystick), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
		img = cv2.putText(img, 'next : ' + str(odom_next), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
		return img

	def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
		if batch_idx==0 and self.current_epoch % 50==0:
			with torch.no_grad():
				anchor_patch, positive_patch, negative_patch, anchor_imu, positive_imu, negative_imu, metadata, anchor_joystick, positive_joystick, negative_joystick = batch
				an_embedding = self.forward(anchor_patch.float().cuda(), anchor_imu.float().cuda(), anchor_joystick.float().cuda())
				pos_embedding = self.forward(positive_patch.float().cuda(), positive_imu.float().cuda(), positive_joystick.float().cuda())
				neg_embedding = self.forward(negative_patch.float().cuda(), negative_imu.float().cuda(), negative_joystick.float().cuda())

			# convert images to opencv format
			anchor_patch = (anchor_patch.cpu().numpy().transpose(0, 2, 3, 1)*255.0).astype(np.uint8)
			# positive_patch = positive_patch.cpu().numpy().transpose(0, 2, 3, 1)
			# negative_patch = negative_patch.cpu().numpy().transpose(0, 2, 3, 1)
			anchor_labelled = []
			for i in range(anchor_patch.shape[0]):
				img = cv2.cvtColor(np.asarray(anchor_patch[i]), cv2.COLOR_RGB2BGR)
				img = cv2.resize(img, (128, 128))

				# write metadata on image
				an_odom_curr = metadata['anchor_curr_odom'][i].numpy()[[0, 2]].round(2)
				an_joystick = metadata['anchor_joystick'][i].numpy().round(2)
				an_odom_next = metadata['anchor_next_odom'][i].numpy()[[0, 2]].round(2)

				img = self.write_metadata_on_image(img, an_odom_curr, an_joystick, an_odom_next)

				anchor_labelled.append(img)
			anchor_patch = np.asarray(anchor_labelled).transpose(0, 3, 1, 2)
			anchor_patch = torch.from_numpy(anchor_patch.astype(np.float32)/255.0).float()

			# positive
			positive_patch = (positive_patch.cpu().numpy().transpose(0, 2, 3, 1)*255.0).astype(np.uint8)
			positive_labelled = []
			for i in range(positive_patch.shape[0]):
				img = cv2.cvtColor(np.asarray(positive_patch[i]), cv2.COLOR_RGB2BGR)
				img = cv2.resize(img, (128, 128))

				an_odom_curr = metadata['positive_curr_odom'][i].numpy()[[0, 2]].round(2)
				an_joystick = metadata['positive_joystick'][i].numpy().round(2)
				an_odom_next = metadata['positive_next_odom'][i].numpy()[[0, 2]].round(2)

				# write metadata on image
				img = self.write_metadata_on_image(img, an_odom_curr, an_joystick, an_odom_next)
				positive_labelled.append(img)
			positive_patch = np.asarray(positive_labelled).transpose(0, 3, 1, 2)
			positive_patch = torch.from_numpy(positive_patch.astype(np.float32)/255.0).float()

			# negative
			negative_patch = (negative_patch.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
			negative_labelled = []
			for i in range(negative_patch.shape[0]):
				img = cv2.cvtColor(np.asarray(negative_patch[i]), cv2.COLOR_RGB2BGR)
				img = cv2.resize(img, (128, 128))

				an_odom_curr = metadata['negative_curr_odom'][i].numpy()[[0, 2]].round(2)
				an_joystick = metadata['negative_joystick'][i].numpy().round(2)
				an_odom_next = metadata['negative_next_odom'][i].numpy()[[0, 2]].round(2)

				# write metadata on image
				img = self.write_metadata_on_image(img, an_odom_curr, an_joystick, an_odom_next)
				negative_labelled.append(img)
			negative_patch = np.asarray(negative_labelled).transpose(0, 3, 1, 2)
			negative_patch = torch.from_numpy(negative_patch.astype(np.float32) / 255.0).float()

			images = torch.cat([anchor_patch, positive_patch, negative_patch], dim=0)
			# images = torch.flip(images, [1])# flip along RGB axis

			embeddings = torch.cat([an_embedding, pos_embedding, neg_embedding], dim=0)

			self.logger.experiment.add_embedding(mat=embeddings,
												 label_img=images,
												 global_step=self.current_epoch)


	def test_step(self, batch, batch_idx):
		return self.validation_step(batch, batch_idx)

	def configure_optimizers(self):
		return torch.optim.AdamW(self.visual_imu_encoder_model.parameters(), lr=3e-4, weight_decay=1e-5)

class TripletDataset(Dataset):
	def __init__(self, data, data_distant_indices):
		self.data = data
		self.data_distant_indices = data_distant_indices

		# process joystick history
		self.data['joystick_1sec_history'] = []
		joystick_history = [[0.0, 0.0] for _ in range(4)]
		for i in range(len(data['joystick'])):
			joystick_history = joystick_history[1:] + [data['joystick'][i]]
			self.data['joystick_1sec_history'].append(joystick_history)

		# invert the weights for the positives (so we sample the easy positives more)
		# for key in list(self.data_distant_indices.keys()):
		# 	self.data_distant_indices[key]['p_weight'] = [1./float(val) for val in self.data_distant_indices[key]['p_weight']]

	def __len__(self):
		return len(list(self.data_distant_indices.keys()))

	def __getitem__(self, idx):
		anchor_idx = list(self.data_distant_indices.keys())[idx]
		positive_idxs = self.data_distant_indices[anchor_idx]['p_idx']
		negative_idxs = self.data_distant_indices[anchor_idx]['n_idx']
		# positive_idx, negative_idx = np.random.choice(positive_idxs), np.random.choice(negative_idxs)

		# sample an easy positive
		positive_idx = random.choice(positive_idxs)
		# sample a hard negative
		negative_idx = random.choice(negative_idxs)

		anchor_patch = random.choice(self.data['patches'][anchor_idx]).transpose(2, 0, 1).astype(np.float32)/255.0
		positive_patch = random.choice(self.data['patches'][positive_idx]).transpose(2, 0, 1).astype(np.float32)/255.0
		negative_patch = random.choice(self.data['patches'][negative_idx]).transpose(2, 0, 1).astype(np.float32)/255.0
		# cv2.imshow('neg', negative_patch)
		# cv2.waitKey(0)

		anchor_accel = np.asarray(self.data['accel_msg'][anchor_idx])
		anchor_gyro = np.asarray(self.data['gyro_msg'][anchor_idx])
		anchor_imu = np.concatenate((anchor_accel, anchor_gyro))

		positive_accel = np.asarray(self.data['accel_msg'][anchor_idx])
		positive_gyro = np.asarray(self.data['gyro_msg'][anchor_idx])
		positive_imu = np.concatenate((positive_accel, positive_gyro))

		negative_accel = np.asarray(self.data['accel_msg'][anchor_idx])
		negative_gyro = np.asarray(self.data['gyro_msg'][anchor_idx])
		negative_imu = np.concatenate((negative_accel, negative_gyro))

		anchor_joystick = np.asarray(self.data['joystick_1sec_history'][anchor_idx]).flatten()
		positive_joystick = np.asarray(self.data['joystick_1sec_history'][positive_idx]).flatten()
		negative_joystick = np.asarray(self.data['joystick_1sec_history'][negative_idx]).flatten()

		metadata = {
			'anchor_curr_odom': np.asarray(self.data['odom'][anchor_idx+4][:3]),
			'anchor_joystick': np.asarray(self.data['joystick'][anchor_idx]),
			'anchor_next_odom': np.asarray(self.data['odom'][anchor_idx+5][:3]),

			'positive_curr_odom': np.asarray(self.data['odom'][positive_idx+4][:3]),
			'positive_joystick': np.asarray(self.data['joystick'][positive_idx]),
			'positive_next_odom': np.asarray(self.data['odom'][positive_idx+5][:3]),

			'negative_curr_odom': np.asarray(self.data['odom'][negative_idx + 4][:3]),
			'negative_joystick': np.asarray(self.data['joystick'][negative_idx]),
			'negative_next_odom': np.asarray(self.data['odom'][negative_idx + 5][:3]),
		}

		return anchor_patch, positive_patch, negative_patch, anchor_imu, positive_imu, negative_imu, metadata, anchor_joystick, positive_joystick, negative_joystick

class TripletDataModule(pl.LightningDataModule):
	def __init__(self, data_dir, train_dataset_names, val_dataset_names, batch_size):
		super(TripletDataModule, self).__init__()
		self.batch_size = batch_size

		# first read the pickle files and setup the datasets
		self.setup_datasets(data_dir, train_dataset_names, val_dataset_names)

	def setup_datasets(self, data_dir, train_dataset_names, val_dataset_names):
		train_datasets = []
		for dataset_name in train_dataset_names:
			data_files = os.listdir(os.path.join(data_dir, dataset_name))
			data_files = [file for file in data_files if file.endswith('data_1.pkl')]
			for file in data_files:
				data = pickle.load(open(os.path.join(data_dir, dataset_name, file), 'rb'))
				data_distant_indices = pickle.load(open(os.path.join(data_dir, dataset_name, file.replace('data_1.pkl', 'distant_indices_abs.pkl')), 'rb'))
				dataset = TripletDataset(data, data_distant_indices)
				if len(dataset) > 0:
					train_datasets.append(dataset)
		self.training_dataset = torch.utils.data.ConcatDataset(train_datasets)
		cprint('Num training datapoints : ' + str(len(self.training_dataset)), 'green', attrs=['bold'])

		val_datasets = []
		for dataset_name in val_dataset_names:
			data_files = os.listdir(os.path.join(data_dir, dataset_name))
			data_files = [file for file in data_files if file.endswith('data_1.pkl')]
			for file in data_files:
				data = pickle.load(open(os.path.join(data_dir, dataset_name, file), 'rb'))
				data_distant_indices = pickle.load(open(os.path.join(data_dir, dataset_name, file.replace('data_1.pkl', 'distant_indices_abs.pkl')), 'rb'))
				dataset = TripletDataset(data, data_distant_indices)
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
	# setup argparse
	parser = argparse.ArgumentParser(description='Train Experience Encoder')
	parser.add_argument('--data_dir', type=str, default='/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/ahg_indoor_bags', help='path to data directory')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
	parser.add_argument('--margin', type=float, default=1.0, help='Number of GPUs to use')
	parser.add_argument('--max_epochs', type=int, default=100000, help='Number of epochs to train for')
	parser.add_argument('--train_dataset_names', nargs='+', default=['train1_data','train3_data','train5_data',
																	 'train7_data', 'train9_data', 'train11_data', 'train13_data', 'train15_data',
																	 'train17_data','train19_data','train21_data','train23_data','train8_data', 'train10_data', 'train12_data'])
	parser.add_argument('--val_dataset_names', nargs='+', default=['train2_data','train4_data','train6_data', 'train14_data', 'train16_data',
																   'train18_data','train20_data','train22_data','train24_data'])
	args = parser.parse_args()

	dm = TripletDataModule(args.data_dir, args.train_dataset_names, args.val_dataset_names, args.batch_size)
	model = EncoderModel(args, margin=args.margin)
	model = model.cuda()

	early_stopping_cb = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00, patience=1000)
	model_checkpoint_cb = ModelCheckpoint(dirpath='models/encoder/',
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






