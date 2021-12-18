import pickle
import cv2
import glob

import numpy as np
from tqdm import tqdm

dataset_path = '/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/ahg_indoor_bags/train1_filter_data'
pickle_file_paths = glob.glob(dataset_path + '/*.pkl')

print(pickle_file_paths)

for pickle_file in tqdm(pickle_file_paths):
	print('pickle_file :', pickle_file)
	with open(pickle_file, 'rb') as f:
		data = pickle.load(f)
		print('keys : ', data.keys())
		print('len of data : ', len(data['patches']))

	for i in range(len(data['patches'])):
		for j in range(len(data['patches'][i])):
			cv2.imshow('patches', data['patches'][i][j])
			cv2.waitKey(0)
