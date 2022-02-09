import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
from termcolor import cprint
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/ahg_indoor_bags')
parser.add_argument('--dataset_names', nargs='+', default=['train13'])
parser.add_argument('--visualize', action='store_true', default=False)
parser.add_argument('--num_threads', type=int, default=10)
args = parser.parse_args()
if args.visualize:
	import cv2

min_threads = min(args.num_threads, len(args.dataset_names))
cprint('Using ' + str(min_threads) + ' threads', 'white', attrs=['bold'])

JOYSTICK_FWD_VEL_THRESHOLD = 0.75
JOYSTICK_ANGULAR_VEL_THRESHOLD = 0.25
ODOM_FWD_VEL_THRESHOLD = 0.75
ODOM_SIDE_VEL_THRESHOLD = 0.25
ODOM_ANGULAR_VEL_THRESHOLD = 0.25

KD_THRESH_FWD_VEL = 0.5
KD_THRESH_ANGULAR_VEL = 0.2

def joysticks_symmetric_close(joystick1, joystick2):
	if abs(abs(joystick1[0]) - abs(joystick2[0])) > JOYSTICK_FWD_VEL_THRESHOLD: return False
	elif abs(abs(joystick1[1]) - abs(joystick2[1])) > JOYSTICK_ANGULAR_VEL_THRESHOLD: return False
	if joystick1[1] * joystick2[1] > 0: return False # same sign
	return True

def odoms_symmetric_close(odom1, odom2):
	if abs(abs(odom1[0]) - abs(odom2[0])) > ODOM_FWD_VEL_THRESHOLD: return False
	if abs(abs(odom1[1]) - abs(odom2[1])) > ODOM_SIDE_VEL_THRESHOLD: return False
	if abs(abs(odom1[2]) - abs(odom2[2])) > ODOM_ANGULAR_VEL_THRESHOLD: return False
	if odom1[1] * odom2[1] > 0: return False # same sign
	if odom1[2] * odom2[2] > 0: return False # same sign
	return True

def joysticks_close(joystick1, joystick2):
	if abs(joystick1[0] - joystick2[0]) > JOYSTICK_FWD_VEL_THRESHOLD: return False
	if abs(joystick1[1] - joystick2[1]) > JOYSTICK_ANGULAR_VEL_THRESHOLD: return False
	if joystick1[1] * joystick2[1] < 0: return False # opposite sign
	return True

def odoms_close(odom1, odom2):
	if abs(odom1[0] - odom2[0]) > ODOM_FWD_VEL_THRESHOLD: return False
	if abs(odom1[1] - odom2[1]) > ODOM_SIDE_VEL_THRESHOLD: return False
	if abs(odom1[2] - odom2[2]) > ODOM_ANGULAR_VEL_THRESHOLD: return False
	if odom1[1] * odom2[1] < 0: return False # opposite sign
	if odom1[2] * odom2[2] < 0: return False # opposite sign
	return True

def find_positive_and_negative_indices(anchor_idx, data):
	anchor_joystick = data['joystick'][anchor_idx]
	anchor_curr_odom = np.asarray(data['odom'][anchor_idx+4])[:3]
	anchor_next_odom = np.asarray(data['odom'][anchor_idx+5])[:3]

	data_len = len(data['odom'])
	distant_patch_idx, identical_patch_idx = [], []
	distant_patch_weight, identical_patch_weight = [], []

	full_list = list(set(range(data_len-6)) - set([anchor_idx]))
	for i in full_list:
		if not data['patches_found'][i]: continue
		joystick = np.asarray(data['joystick'][i])
		curr_odom = np.asarray(data['odom'][i+4])[:3]
		next_odom = np.asarray(data['odom'][i+5])[:3]

		# normal situation where all values including signs are close together
		if joysticks_close(anchor_joystick, joystick) and \
			odoms_close(curr_odom, anchor_curr_odom) and \
			odoms_close(next_odom, anchor_next_odom):

			identical_patch_idx.append(i)
			weight = np.linalg.norm(anchor_joystick-joystick) + \
					 np.linalg.norm(curr_odom-anchor_curr_odom) + \
					 np.linalg.norm(next_odom-anchor_next_odom)
			identical_patch_weight.append(weight)

		# perfect symmetric situation
		elif joysticks_symmetric_close(anchor_joystick, joystick) and \
			odoms_symmetric_close(curr_odom, anchor_curr_odom) and \
			odoms_symmetric_close(next_odom, anchor_next_odom):
			identical_patch_idx.append(i)
			weight = np.linalg.norm(anchor_joystick - joystick) + \
					 np.linalg.norm(curr_odom - anchor_curr_odom) + \
					 np.linalg.norm(next_odom - anchor_next_odom)
			identical_patch_weight.append(weight)

		# did not match at all
		elif not odoms_close(curr_odom, anchor_curr_odom):
			distant_patch_idx.append(i)
			weight = np.linalg.norm(anchor_joystick - joystick) + \
					 np.linalg.norm(curr_odom - anchor_curr_odom) + \
					 np.linalg.norm(next_odom - anchor_next_odom)
			distant_patch_weight.append(weight)

	return identical_patch_idx, identical_patch_weight, distant_patch_idx, distant_patch_weight

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

def process_datset(dataset):
	data_file = os.path.join(args.data_dir, dataset + '_data', 'data_1.pkl')
	data = pickle.load(open(data_file, 'rb'))
	data_len = len(data['odom'])
	print('\nprocessing file : ', data_file, '\n')

	valid_data = {}
	for i in tqdm(range(data_len - 7)):
		# if there were no patch data, skip
		if not data['patches_found'][i]: continue

		positive_patch_idx, positive_patch_weight, distant_patch_idx, distant_patch_weight = find_positive_and_negative_indices_v2(
			i, data)

		# only use this data sample if it has atleast 1 distant patch and 1 positive patch
		if len(distant_patch_idx) < 1 or len(positive_patch_idx) < 1: continue

		# print('len positive : ', len(positive_patch_idx))
		# print('len distant : ', len(distant_patch_idx))
		# input('press enter to continue...')

		# valid anchor and negative patch found !
		valid_data[i] = {'p_idx': positive_patch_idx,
						 'p_weight': positive_patch_weight,
						 'n_idx': distant_patch_idx,
						 'n_weight': distant_patch_weight}

		if args.visualize:
			patches = data['patches'][i]
			cv2.imshow('anchor', patches[0])
			print('anchor properties -- ')
			print('curr odom', data['odom'][i + 4][:3])
			print('joystick', data['joystick'][i])
			print('next odom', data['odom'][i + 5][:3])

			cv2.imshow('neg', data['patches'][distant_patch_idx[0]][0])
			print('neg properties -- ')
			print('curr odom', data['odom'][distant_patch_idx[0] + 4][:3])
			print('joystick', data['joystick'][distant_patch_idx[0]])
			print('next odom', data['odom'][distant_patch_idx[0] + 5][:3])

			cv2.imshow('pos', data['patches'][positive_patch_idx[0]][0])
			print('pos properties -- ')
			print('curr odom', data['odom'][positive_patch_idx[0] + 4][:3])
			print('joystick', data['joystick'][positive_patch_idx[0]])
			print('next odom', data['odom'][positive_patch_idx[0] + 5][:3])

			# numpy checks
			# anchor_curr_odom = np.asarray(data['odom'][i + 4])[:3]
			# anchor_next_odom = np.asarray(data['odom'][i + 5])[:3]
			# positive_odom_curr = np.asarray(data['odom'][positive_patch_idx[0] + 4])[:3]
			# positive_odom_next = np.asarray(data['odom'][positive_patch_idx[0] + 5])[:3]
			# negative_odom_curr = np.asarray(data['odom'][distant_patch_idx[0] + 4])[:3]
			# negative_odom_next = np.asarray(data['odom'][distant_patch_idx[0] + 5])[:3]
			# print('positive : ', np.allclose(positive_odom_curr, anchor_curr_odom, 0.1))
			# print('negative : ', np.allclose(negative_odom_curr, anchor_curr_odom, 0.1))
			# print('positive : ', np.allclose(positive_odom_next, anchor_next_odom, 0.02))
			# print('negative : ', np.allclose(negative_odom_next, anchor_next_odom, 0.02))
			cv2.waitKey(0)

	# remove negative patches that are not an anchor
	cprint('\nRemoving negatives that are not an anchor !\n', 'green', attrs=['bold'])
	anchors = list(valid_data.keys())
	for valid_anchor in anchors:
		valid_negatives, valid_negatives_weights = [], []
		all_negatives = valid_data[valid_anchor]['n_idx']
		all_negatives_weights = valid_data[valid_anchor]['n_weight']

		for i, val in enumerate(all_negatives):
			if val in anchors:
				valid_negatives.append(all_negatives[i])
				valid_negatives_weights.append(all_negatives_weights[i])

		valid_data[valid_anchor]['n_idx'] = valid_negatives
		valid_data[valid_anchor]['n_weight'] = valid_negatives_weights

		if len(valid_negatives) == 0:
			raise Exception("Removed all negatives for this sample!! :(")

	# order the positives and negatives based on their weights
	cprint('\nOrdering positives and negatives based on their weights !\n', 'green', attrs=['bold'])
	for valid_anchor in anchors:
		positive_idx = valid_data[valid_anchor]['p_idx']
		negative_idx = valid_data[valid_anchor]['n_idx']

		positive_weight = valid_data[valid_anchor]['p_weight']
		negative_weight = valid_data[valid_anchor]['n_weight']

		valid_data[valid_anchor]['p_idx'] = [positive_idx[i] for i in np.argsort(positive_weight)]
		valid_data[valid_anchor]['n_idx'] = [negative_idx[i] for i in np.argsort(negative_weight)]
		valid_data[valid_anchor]['p_weight'] = np.sort(positive_weight)
		valid_data[valid_anchor]['n_weight'] = np.sort(negative_weight)

	# delete weight from anchor_data
	# del valid_data[valid_anchor]['p_weight']
	# del valid_data[valid_anchor]['n_weight']

	print('total samples : ', data_len)
	print('total valid samples : ', len(list(valid_data.keys())))
	# save valid_data dictionary as a json file
	pickle.dump(valid_data, open(data_file.replace('data_1.pkl', 'distant_indices_abs.pkl'), 'wb'))
	cprint('Saved distant indices to file', 'green', attrs=['bold'])

thread_list = []
for dataset in tqdm(args.dataset_names):
	if len(thread_list) < min_threads:
		print('\nCreating thread for dataset : ', dataset)
		t = threading.Thread(target=process_datset, args=(dataset,))
		thread_list.append(t)
		t.start()


	# wait for all threads to complete
	if len(thread_list) == min_threads:
		for t in thread_list:
			t.join()
		thread_list = []




