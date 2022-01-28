import pickle
import cv2
import glob

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

dataset_path = '/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/ahg_indoor_bags/train1_data/'
pickle_file_paths = glob.glob(dataset_path + '/*.pkl')

print(pickle_file_paths)

for pickle_file in tqdm(pickle_file_paths):
	print('pickle_file :', pickle_file)
	with open(pickle_file, 'rb') as f:
		data = pickle.load(f)
		print('keys : ', data.keys())
		print('len of data : ', len(data['patches']))

	# for i in range(len(data['patches'])):
	# 	print('i : ', i)
	# 	for j in range(len(data['patches'][i])):
	# 		cv2.imshow('patches', data['patches'][i][j])
	# 		cv2.waitKey(0)
	#
	# 	print('accel shape : ', data['accel'][i].shape)
	# 	print('gyro shape : ', data['gyro'][i].shape)
	# 	print('odom shape : ', data['odom'][i].shape)

	print(len(data['joystick']))

	data['joystick'] = np.array(data['joystick'])
	data['odom'] = np.array(data['odom'])



	plt.subplot(2, 1, 1)
	# data['joystick'][:, 0] = savgol_filter(data['joystick'][:, 0], 19, 3)
	plt.plot(np.arange(len(data['joystick'][:-5, 0])), data['joystick'][:-5, 0])
	# plt.plot(np.arange(len(data['joystick'][:, 0])), savgol_filter(data['joystick'][:, 0], 19, 3), 'r')
	# plt.plot(np.arange(len(data['vescdrive'][:, 0])), data['vescdrive'][:, 0])
	plt.plot(np.arange(len(data['odom'][5:, 0])), data['odom'][5:, 0])
	plt.xlim(0, 1000)

	plt.subplot(2, 1, 2)
	plt.plot(np.arange(len(data['joystick'][:-5, 1])), data['joystick'][:-5, 1])
	# plt.plot(np.arange(len(data['joystick'][:, 1])), savgol_filter(data['joystick'][:, 1], 19, 3), 'r')
	# plt.plot(np.arange(len(data['vescdrive'][:, 1])), data['vescdrive'][:, 1])
	plt.plot(np.arange(len(data['odom'][5:, 0])), data['odom'][5:, 2])

	plt.xlim(0, 1000)
	plt.show()



	# data['odom'][:, 0] = np.sqrt(data['odom'][:, 3] ** 2 + data['odom'][:, 4] ** 2)

	# plt.figure(figsize=(20, 10))
	# plt.subplot(2, 1, 1)
	# plt.plot(data['odom'][:, 0], '-r')
	# plt.plot(data['joystick'][:, 0], '-b')
	# plt.subplot(2, 1, 2)
	# plt.plot(data['odom'][:, 2], '-r')
	# plt.plot(data['joystick'][:, 1], '-b')
	# plt.plot(data['vescdrive'][:, 1], '-g')
	# plt.show()

	# for i in range(len(data['patches'])):
	# 	cv2.imshow('patch 1', data['patches'][i][-1])
	# 	print('patch found : ', data['patches_found'][i])
	# 	cv2.waitKey(0)