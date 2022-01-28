import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
from termcolor import cprint

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/ahg_indoor_bags')
parser.add_argument('--dataset_names', nargs='+', default=['train3',
														   'train4',
														   'train5',
														   'train6',
														   'train7',
														   'train8',
														   'train9',
														   'train10'])
parser.add_argument('--visualize', action='store_true', default=False)
args = parser.parse_args()
if args.visualize:
	import cv2

triplets = []

def find_positive_and_negative_indices(anchor_idx, data):
	anchor_joystick = data['joystick'][anchor_idx]
	anchor_curr_odom = np.asarray(data['odom'][anchor_idx+4])[:3]
	anchor_next_odom = np.asarray(data['odom'][anchor_idx+5])[:3]
	data_len = len(data['odom'])
	distant_patch_idx, identical_patch_idx = [], []

	full_list = list(set(range(data_len-6)) - set([anchor_idx]))
	for i in full_list:
		joystick = np.asarray(data['joystick'][i])
		curr_odom = np.asarray(data['odom'][i+4])[:3]
		next_odom = np.asarray(data['odom'][i+5])[:3]

		# patches are distant if they have the *same* joystick and input odom but different odom responses...
		DIST_ODOM_THRESHOLD = 0.2
		IDENTICAL_ODOM_THRESHOLD = 0.02
		if np.allclose(anchor_joystick, joystick, 1e-2) and np.allclose(curr_odom, anchor_curr_odom, 0.1):
			if np.linalg.norm(next_odom - anchor_next_odom) > DIST_ODOM_THRESHOLD:
				distant_patch_idx.append(i)
			elif np.allclose(next_odom, anchor_next_odom, IDENTICAL_ODOM_THRESHOLD):
				identical_patch_idx.append(i)

	return identical_patch_idx, distant_patch_idx

for dataset in args.dataset_names:
	data_file = os.path.join(args.data_dir, dataset + '_data', 'data_1.pkl')
	data = pickle.load(open(data_file, 'rb'))
	data_len = len(data['odom'])
	print('processing file : ', data_file)

	valid_data = {}
	for i in tqdm(range(data_len-7)):
		odom = data['odom'][i]
		patches = data['patches'][i]

		positive_patch_idx, distant_patch_idx = find_positive_and_negative_indices(i, data)

		# only use this data sample if it has atleast 1 distant patch and 1 positive patch
		if len(distant_patch_idx) < 1 or len(positive_patch_idx) < 1: continue

		# valid anchor and negative patch found !
		valid_data[i] = {'p_idx': positive_patch_idx, 'n_idx': distant_patch_idx}

		if args.visualize:
			cv2.imshow('anchor', patches[0])
			print('anchor properties -- ')
			print('curr odom', data['odom'][i+4][:3])
			print('joystick', data['joystick'][i])
			print('next odom', data['odom'][i+5][:3])

			cv2.imshow('neg', data['patches'][distant_patch_idx[0]][0])
			print('neg properties -- ')
			print('curr odom', data['odom'][distant_patch_idx[0]+4][:3])
			print('joystick', data['joystick'][distant_patch_idx[0]])
			print('next odom', data['odom'][distant_patch_idx[0]+5][:3])

			cv2.imshow('pos', data['patches'][positive_patch_idx[0]][0])
			print('pos properties -- ')
			print('curr odom', data['odom'][positive_patch_idx[0] + 4][:3])
			print('joystick', data['joystick'][positive_patch_idx[0]])
			print('next odom', data['odom'][positive_patch_idx[0] + 5][:3])

			# numpy checks
			anchor_curr_odom = np.asarray(data['odom'][i+4])[:3]
			anchor_next_odom = np.asarray(data['odom'][i+5])[:3]
			positive_odom_curr = np.asarray(data['odom'][positive_patch_idx[0] + 4])[:3]
			positive_odom_next = np.asarray(data['odom'][positive_patch_idx[0] + 5])[:3]
			negative_odom_curr = np.asarray(data['odom'][distant_patch_idx[0] + 4])[:3]
			negative_odom_next = np.asarray(data['odom'][distant_patch_idx[0] + 5])[:3]
			print('positive : ', np.allclose(positive_odom_curr, anchor_curr_odom, 0.1))
			print('negative : ', np.allclose(negative_odom_curr, anchor_curr_odom, 0.1))
			print('positive : ', np.allclose(positive_odom_next, anchor_next_odom, 0.02))
			print('negative : ', np.allclose(negative_odom_next, anchor_next_odom, 0.02))
			cv2.waitKey(0)

	print(valid_data)
	print('total samples : ', data_len)
	print('total valid samples : ', len(list(valid_data.keys())))
	# save valid_data dictionary as a json file
	pickle.dump(valid_data, open(data_file.replace('data_1.pkl', 'distant_indices.pkl'), 'wb'))
	cprint('Saved distant indices to file', 'green', attrs=['bold'])



