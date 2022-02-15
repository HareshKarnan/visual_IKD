import argparse
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from termcolor import cprint
import threading
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/ahg_indoor_bags')
parser.add_argument('--dataset_names', nargs='+', default=['train2'])
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

KD_THRESH_FWD_VEL = 0.1
KD_THRESH_ANGULAR_VEL = 0.1

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
	if abs(abs(kd_response_1[0]) - abs(kd_response_2[0])) > KD_THRESH_FWD_VEL: return False
	if abs(abs(kd_response_1[1]) - abs(kd_response_2[1])) > KD_THRESH_ANGULAR_VEL: return False

	# response was same then..
	return True



if __name__ == '__main__':
	for dataset in tqdm(args.dataset_names):
		data_file = os.path.join(args.data_dir, dataset + '_data', 'data_1.pkl')
		data = pickle.load(open(data_file, 'rb'))
		data_len = len(data['odom'])
		print('\nprocessing file : ', data_file, '\n')

		valid_data = {}
		data_indices, kd_response_list, fwd_vel = [], [], []
		for i in tqdm(range(data_len - 7)):
			# if there were no patch data, skip
			if not data['patches_found'][i]: continue

			joystick = data['joystick'][i]
			curr_odom = np.asarray(data['odom'][i + 4])[:3]
			next_odom = np.asarray(data['odom'][i + 5])[:3]

			kd_response = [joystick[0] - next_odom[0],
						   abs(joystick[1]) - abs(next_odom[2])]

			data_indices.append(i)
			kd_response_list.append(kd_response)
			fwd_vel.append(curr_odom[0])

	# now run kmeans clustering
	print(i)

	print(kd_response_list)
	kd_response_list = np.asarray(kd_response_list)


	# distortions = []
	# for num_clusters in range(2, 30, 2):
	# 	kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(kd_response_list)
	# 	distortions.append(kmeans.inertia_)
	#
	# plt.plot(np.asarray(range(2, 30, 2)), np.asarray(distortions))
	# plt.show()

	# kmeans = KMeans(n_clusters=4, random_state=0).fit(kd_response_list)
	# kmeans_label = kmeans.predict(kd_response_list)

	plt.figure(figsize=(20, 20))
	# plt.scatter(kd_response_list[:, 0], kd_response_list[:, 1], c=kmeans_label, cmap='viridis', s=100)
	plt.scatter(kd_response_list[:, 0], kd_response_list[:, 1], c=fwd_vel, cmap='Greens', s=100)
	plt.xlim([-3, 3])
	plt.ylim([-2, 3])
	plt.xlabel('Forward vel KD response')
	plt.ylabel('Angular vel KD response')
	plt.show()




