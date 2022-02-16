import pickle
import cv2
import glob

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

dataset_path = '/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/outdoor_bags/train6_data/'
pickle_file_paths = glob.glob(dataset_path + '/*data_1.pkl')

print(pickle_file_paths)

DELAY_THRESHOLD = 5

for pickle_file in tqdm(pickle_file_paths):
	print('pickle_file :', pickle_file)
	with open(pickle_file, 'rb') as f:
		data = pickle.load(f)
		print('keys : ', data.keys())
		print('len of data : ', len(data['patches']))

	print(len(data['joystick']))

	data['joystick'] = np.array(data['joystick'])
	data['odom'] = np.array(data['odom'])

	plt.subplot(2, 1, 1)
	plt.plot(np.arange(len(data['joystick'][:-DELAY_THRESHOLD, 0])), data['joystick'][:-DELAY_THRESHOLD, 0])
	plt.plot(np.arange(len(data['odom'][DELAY_THRESHOLD:, 0])), data['odom'][DELAY_THRESHOLD:, 0])
	plt.xlim(0, 1000)

	plt.subplot(2, 1, 2)
	plt.plot(np.arange(len(data['joystick'][:-DELAY_THRESHOLD, 1])), data['joystick'][:-DELAY_THRESHOLD, 1])
	plt.plot(np.arange(len(data['odom'][DELAY_THRESHOLD:, 0])), data['odom'][DELAY_THRESHOLD:, 2])

	plt.xlim(0, 1000)
	plt.show()

	errors_v, errors_w = [], []
	for i in range(1, 20):
		errors_v.append(np.linalg.norm(data['joystick'][:-i, 0] - data['odom'][i:, 0]))
		errors_w.append(np.linalg.norm(data['joystick'][:-i, 1] - data['odom'][i:, 2]))

	plt.xticks(np.arange(1, 20, step=1.0))
	plt.plot(errors_v)
	plt.show()
	plt.xticks(np.arange(1, 20, step=1.0))
	plt.plot(errors_w)
	plt.show()