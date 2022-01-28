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

class VisualIMUEncoder(nn.Module):
	def __init__(self):
		super(VisualIMUEncoder, self).__init__()
		self.visual_encoder = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, stride=2),
			nn.BatchNorm2d(16), nn.PReLU(),  # 31x31
			nn.MaxPool2d(kernel_size=3, stride=2),  # 15x15
			nn.Conv2d(16, 32, kernel_size=3, stride=2),
			nn.BatchNorm2d(32), nn.PReLU(),  # 7x7
			nn.MaxPool2d(kernel_size=3, stride=2),  # 3x3
			nn.Flatten(),
			nn.Linear(3 * 3 * 32, 64), nn.BatchNorm1d(64), nn.PReLU(),
			nn.Linear(64, 32)
		)

		self.imu_net = nn.Sequential(
			nn.Linear(200 * 3 + 60 * 3, 128), nn.BatchNorm1d(128), nn.PReLU(),
			nn.Linear(128, 64), nn.BatchNorm1d(64), nn.PReLU(),
			nn.Linear(64, 16),
		)

		self.imu_visual_net = nn.Sequential(
			nn.Linear(32 + 16, 32), nn.BatchNorm1d(32), nn.PReLU(),
			nn.Linear(32, 6),
		)

	def forward(self, visual_input, imu_input):
		patch_embedding = self.visual_encoder(visual_input)
		imu_embedding = self.imu_net(imu_input)
		embedding = self.imu_visual_net(torch.cat([patch_embedding, imu_embedding], dim=1))
		embedding = F.normalize(embedding, p=2, dim=1)
		return embedding

class EncoderModel(pl.LightningModule):
	def __init__(self, args):
		super(EncoderModel, self).__init__()
		self.save_hyperparameters('args')
		self.visual_imu_encoder_model = VisualIMUEncoder()
		self.loss = nn.TripletMarginLoss(margin=1.0, swap=True)

	def forward(self, patch, imu):
		return self.visual_imu_encoder_model(patch, imu)

	def training_step(self, batch, batch_idx):
		anchor_patch, positive_patch, negative_patch, anchor_imu, positive_imu, negative_imu = batch
		an_embedding = self.forward(anchor_patch.float(), anchor_imu.float())
		pos_embedding = self.forward(positive_patch.float(), positive_imu.float())
		neg_embedding = self.forward(negative_patch.float(), negative_imu.float())
		loss = self.loss(an_embedding, pos_embedding, neg_embedding)
		self.log('train_loss', loss, prog_bar=True, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		anchor_patch, positive_patch, negative_patch, anchor_imu, positive_imu, negative_imu = batch
		an_embedding = self.forward(anchor_patch.float(), anchor_imu.float())
		pos_embedding = self.forward(positive_patch.float(), positive_imu.float())
		neg_embedding = self.forward(negative_patch.float(), negative_imu.float())
		loss = self.loss(an_embedding, pos_embedding, neg_embedding)
		self.log('val_loss', loss, prog_bar=True, logger=True)
		return loss

	def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
		if batch_idx==0:
			with torch.no_grad():
				anchor_patch, positive_patch, negative_patch, anchor_imu, positive_imu, negative_imu = batch
				an_embedding = self.forward(anchor_patch.float().cuda(), anchor_imu.float().cuda())
				pos_embedding = self.forward(positive_patch.float().cuda(), positive_imu.float().cuda())
				neg_embedding = self.forward(negative_patch.float().cuda(), negative_imu.float().cuda())

			images = torch.cat([anchor_patch, positive_patch, negative_patch], dim=0)
			embeddings = torch.cat([an_embedding, pos_embedding, neg_embedding], dim=0)
			labels = ['P' for _ in range(2*an_embedding.shape[0])]
			labels.extend(['N' for _ in range(neg_embedding.shape[0])])

			self.logger.experiment.add_embedding(mat=embeddings, label_img=images,
												 global_step=self.current_epoch, metadata=labels)


	def test_step(self, batch, batch_idx):
		return self.validation_step(batch, batch_idx)

	def configure_optimizers(self):
		return torch.optim.AdamW(self.visual_imu_encoder_model.parameters(), lr=3e-5, weight_decay=1e-6)

class TripletDataset(Dataset):
	def __init__(self, data, data_distant_indices):
		self.data = data
		self.data_distant_indices = data_distant_indices

	def __len__(self):
		return len(list(self.data_distant_indices.keys()))

	def __getitem__(self, idx):
		anchor_idx = list(self.data_distant_indices.keys())[idx]
		positive_idxs = self.data_distant_indices[anchor_idx]['n_idx']
		negative_idxs = self.data_distant_indices[anchor_idx]['p_idx']
		positive_idx, negative_idx = np.random.choice(positive_idxs), np.random.choice(negative_idxs)

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

		return anchor_patch, positive_patch, negative_patch, anchor_imu, positive_imu, negative_imu

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
				data_distant_indices = pickle.load(open(os.path.join(data_dir, dataset_name, file.replace('data_1.pkl', 'distant_indices.pkl')), 'rb'))
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
				data_distant_indices = pickle.load(open(os.path.join(data_dir, dataset_name, file.replace('data_1.pkl', 'distant_indices.pkl')), 'rb'))
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
	parser.add_argument('--batch_size', type=int, default=128, help='batch size')
	parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
	parser.add_argument('--max_epochs', type=int, default=1000, help='Number of epochs to train for')
	parser.add_argument('--train_dataset_names', nargs='+', default=['train1_data'])
	parser.add_argument('--val_dataset_names', nargs='+', default=['train2_data'])
	args = parser.parse_args()

	dm = TripletDataModule(args.data_dir, args.train_dataset_names, args.val_dataset_names, args.batch_size)
	model = EncoderModel(args)
	model = model.cuda()

	early_stopping_cb = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00, patience=100)
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






