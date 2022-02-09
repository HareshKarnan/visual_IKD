import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
from termcolor import cprint
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/ahg_indoor_bags')
parser.add_argument('--dataset_names', nargs='+', default=['train1', 'train2', 'train3', 'train4', 'train5', 'train6', 'train7', 'train8', 'train9', 'train10','train11', 'train12', 'train17', 'train18', 'train19', 'train20', 'train21', 'train22', 'train23'])
parser.add_argument('--visualize', action='store_true', default=False)
parser.add_argument('--num_threads', type=int, default=10)
args = parser.parse_args()
if args.visualize:
	import cv2

min_threads = min(args.num_threads, len(args.dataset_names))
cprint('Using ' + str(min_threads) + ' threads', 'white', attrs=['bold'])

KD_THRESH_FWD_VEL = 0.5
KD_THRESH_ANGULAR_VEL = 0.2


def find_positive_and_negative_indices_v2(anchor_idx, data):
	anchor_joystick = data['joystick'][anchor_idx]
	anchor_curr_odom = np.asarray(data['odom'][anchor_idx+4])[:3]
	anchor_next_odom = np.asarray(data['odom'][anchor_idx+5])[:3]

	anchor_kd_response = [anchor_joystick[0] - anchor_next_odom[0],
						  anchor_joystick[1] - anchor_next_odom[2]]

	data_len = len(data['odom'])
	distant_patch_idx, identical_patch_idx = [], []
	distant_patch_weight, identical_patch_weight = [], []

	full_list = list(set(range(data_len-6)) - set([anchor_idx]))
	for i in full_list:
		if not data['patches_found'][i]: continue
		joystick = np.asarray(data['joystick'][i])
		curr_odom = np.asarray(data['odom'][i+4])[:3]
		next_odom = np.asarray(data['odom'][i+5])[:3]

		kd_response = [joystick[0] - next_odom[0],
					   abs(joystick[1]) - abs(next_odom[2])]

		if similar_kd_response(anchor_kd_response, kd_response):
			identical_patch_idx.append(i)
			weight = np.linalg.norm(anchor_joystick - joystick) + \
					 np.linalg.norm(curr_odom - anchor_curr_odom) + \
					 np.linalg.norm(next_odom - anchor_next_odom)
			identical_patch_weight.append(weight)

		else:
			distant_patch_idx.append(i)
			weight = np.linalg.norm(anchor_joystick - joystick) + \
					 np.linalg.norm(curr_odom - anchor_curr_odom) + \
					 np.linalg.norm(next_odom - anchor_next_odom)
			distant_patch_weight.append(weight)

	return identical_patch_idx, identical_patch_weight, distant_patch_idx, distant_patch_weight

def similar_kd_response(kd_response_1, kd_response_2):
	# if signs are different, then response is different
	if kd_response_1[0] * kd_response_2[0] < 0: return False
	if kd_response_1[1] * kd_response_2[1] < 0: return False

	# if magnitude of response is greater than a threshold, then they are different
	if abs(kd_response_1[0] - kd_response_2[0]) > KD_THRESH_FWD_VEL: return False
	if abs(kd_response_1[1] - kd_response_2[0]) > KD_THRESH_ANGULAR_VEL: return False

	# response was same then..
	return True

def process_dataset(dataset):
	data_file = os.path.join(args.data_dir, dataset + '_data', 'data_1.pkl')
	data = pickle.load(open(data_file, 'rb'))
	data_len = len(data['odom'])
	print('\nprocessing file : ', data_file, ' of length : ', data_len, '\n')

	print('keys found : ', data.keys())

	data_indices, kd_response_list = [], []
	for i in tqdm(range(data_len - 7)):
		# if there were no patches found in this frame, then skip
		if not data['patches_found'][i]: continue

		# find KD response
		joystick = np.asarray(data['joystick'][i])
		curr_odom = np.asarray(data['odom'][i+4])[:3]
		next_odom = np.asarray(data['odom'][i+5])[:3]
		kd_response = [joystick[0] - next_odom[0],
					   abs(joystick[1]) - abs(next_odom[2])]

		data_indices.append(i)
		kd_response_list.append(kd_response)

	kd_response_list = np.asarray(kd_response_list)
	data_indices = np.asarray(data_indices)

	# find distance between each pair of patches
	cprint('\nFinding distance between each pair of patches...', 'white', attrs=['bold'])
	distances = np.zeros((len(data_indices), len(data_indices)))
	for i in range(len(data_indices)):
		for j in range(i+1, len(data_indices)):
			dist = np.linalg.norm(kd_response_list[i] - kd_response_list[j])
			distances[i][j] = dist
			distances[j][i] = dist
	cprint('Done', 'white', attrs=['bold'])

	valid_data = {}
	for i, anchor_idx in enumerate(data_indices):
		sorted_distance_indices = np.argsort(distances[i])
		positives_idx = data_indices[sorted_distance_indices[:21]][1:]
		negatives_idx = data_indices[sorted_distance_indices[-50:]]

		valid_data[anchor_idx] = {
			'p_idx': positives_idx,
			'n_idx': negatives_idx
		}

	pickle.dump(valid_data, open(data_file.replace('data_1.pkl', 'distant_indices_abs.pkl'), 'wb'))
	cprint('Processed data file ' + str(data_file), 'green', attrs=['bold'])


thread_list = []
for dataset in tqdm(args.dataset_names):
	if len(thread_list) < min_threads:
		print('\nCreating thread for dataset : ', dataset)
		t = threading.Thread(target=process_dataset, args=(dataset,))
		thread_list.append(t)
		t.start()

	# wait for all threads to complete
	if len(thread_list) == min_threads:
		for t in thread_list:
			t.join()
		thread_list = []




