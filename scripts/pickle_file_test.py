import pickle
import cv2
import glob

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

dataset_path = '/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/ahg_indoor_bags/sample_data/'
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
	print(len(data['vescdrive']))

	data['joystick'] = np.array(data['joystick'])
	data['vescdrive'] = np.array(data['vescdrive'])
	data['odom'] = np.array(data['odom'])


	# plt.subplot(2, 1, 1)
	# plt.plot(np.arange(len(data['joystick'][:, 0])), data['joystick'][:, 0])
	# plt.plot(np.arange(len(data['vescdrive'][:, 0])), data['vescdrive'][:, 0])
	# plt.subplot(2, 1, 2)
	# plt.plot(np.arange(len(data['joystick'][:, 1])), data['joystick'][:, 1])
	# plt.plot(np.arange(len(data['vescdrive'][:, 1])), data['vescdrive'][:, 1])
	# plt.show()
	# data['odom'][:, 0] = np.sqrt(data['odom'][:, 0] ** 2 + data['odom'][:, 1] ** 2)

	plt.plot(data['odom'][:, 0], '-r')
	plt.plot(data['vescdrive'][:, 0], '-b')
	plt.show()
