import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import rosbag
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
import yaml, pickle

def extract_pose(bagfile):
	bag = rosbag.Bag(bagfile)
	# read bag file
	loc_x, loc_y = [], []
	for topic, msg, t in bag.read_messages(topics=['/localization']):
		loc_x.append(msg.pose.x)
		loc_y.append(msg.pose.y)
	bag.close()
	return loc_x, loc_y

def hausdorff_dist(x, y):
	return min(directed_hausdorff(x, y)[0], directed_hausdorff(y, x)[0])

def load_waypoints(waypoint_path):
	# get the ground truth waypoint data
	with open(waypoint_path) as f:
		waypoints = yaml.load(f, Loader=yaml.FullLoader)

	pos = []
	for key in waypoints.keys():
		pos.append(waypoints[key]['position'])
	pos = np.asarray(pos)
	return pos

# ROOT_FOLDER = "/media/haresh/HARESH_DRIV/indoor_trials_trajs"
#
# ground_truth_x, ground_truth_y = extract_pose('data/indoor_trials_trajs/graph_nav_slow.bag')
# ground_truth_traj = np.hstack((ground_truth_x, ground_truth_y)).reshape((-1, 2))
#
# # ground_truth_traj = load_waypoints('waypoints.yaml')
#
# hausdorff_dict_vision = {}
# hausdorff_dict_imu = {}
# hausdorff_dict_rhc = {}
#
# speeds = ['2_0', '2_5', '2_6', '2_7', '2_8', '2_9', "3_0", "3_1", "3_2"]
# # speeds = ['2_9', "3_0", "3_1", "3_2"]
#
# for speed in tqdm(speeds):
# 	if speed not in hausdorff_dict_imu.keys():
# 		hausdorff_dict_imu[speed] = []
# 		hausdorff_dict_vision[speed] = []
# 		hausdorff_dict_rhc[speed] = []
#
# 	# Vision
# 	bagfiles = glob.glob(ROOT_FOLDER + '/vision/' + str(speed)+ '/*.bag')
# 	for i, bagfile in enumerate(bagfiles):
# 		loc_x, loc_y = extract_pose(bagfile)
# 		loc_traj = np.hstack((loc_x, loc_y)).reshape((-1, 2))
# 		hausdorff = hausdorff_dist(ground_truth_traj, loc_traj)
# 		hausdorff_dict_vision[speed].append(hausdorff)
# 		# if speed == '3_1':
# 		# 	print(bagfile, ' ', hausdorff)
#
# 	# IMU
# 	bagfiles = glob.glob(ROOT_FOLDER + '/imu/' + str(speed)+ '/*.bag')
# 	for i, bagfile in enumerate(bagfiles):
# 		loc_x, loc_y = extract_pose(bagfile)
# 		loc_traj = np.hstack((loc_x, loc_y)).reshape((-1, 2))
# 		hausdorff = hausdorff_dist(ground_truth_traj, loc_traj)
# 		hausdorff_dict_imu[speed].append(hausdorff)
# 		if speed == '3_1':
# 			print(bagfile, ' ', hausdorff)
#
# 	# RHC
# 	bagfiles = glob.glob(ROOT_FOLDER + '/rhc/' + str(speed)+ '/*.bag')
# 	for i, bagfile in enumerate(bagfiles):
# 		loc_x, loc_y = extract_pose(bagfile)
# 		loc_traj = np.hstack((loc_x, loc_y)).reshape((-1, 2))
# 		hausdorff = hausdorff_dist(ground_truth_traj, loc_traj)
# 		hausdorff_dict_rhc[speed].append(hausdorff)
# 		# if speed == '3_2':
# 		# 	print(bagfile, ' ', hausdorff)
# 		# if speed == '2_6':
# 		# 	print(bagfile, ' ', hausdorff)
# 		# if speed == '2_9':
# 		# 	print(bagfile, ' ', hausdorff)
#
# print(hausdorff_dict_imu)
#
# # hausdorff_vals = [hausdorff_dict[key] for key in ['2_0', '2_5', '2_6', '2_7', '2_8', '2_9']]
# # hausdorff_vals = [hausdorff_dict_imu[key] for key in ['2_0']]
#
# for key in hausdorff_dict_imu.keys():
# 	print(key)
# 	print('IMU : ')
# 	print('mean : ', np.mean(hausdorff_dict_imu[key]), ' std : ', np.std(hausdorff_dict_imu[key]))
# 	print('Vision : ')
# 	print('mean : ', np.mean(hausdorff_dict_vision[key]), ' std : ', np.std(hausdorff_dict_vision[key]))
# 	print('RHC : ')
# 	print('mean : ', np.mean(hausdorff_dict_rhc[key]), ' std : ', np.std(hausdorff_dict_rhc[key]))
#
# for key in hausdorff_dict_imu.keys():
# 	hausdorff_dict_imu[key] = [np.mean(hausdorff_dict_imu[key]), np.std(hausdorff_dict_imu[key])/len(hausdorff_dict_imu[key])]
# 	hausdorff_dict_vision[key] = [np.mean(hausdorff_dict_vision[key]), np.std(hausdorff_dict_vision[key])/len(hausdorff_dict_imu[key])]
# 	hausdorff_dict_rhc[key] = [np.mean(hausdorff_dict_rhc[key]), np.std(hausdorff_dict_rhc[key])/len(hausdorff_dict_imu[key])]
#
# hausdorff_vals_to_save = {
# 	'imu': hausdorff_dict_imu,
# 	'vision': hausdorff_dict_vision,
# 	'rhc': hausdorff_dict_rhc
# }
#
# # save hausdorff vals into a pickle file
# with open('data/hausdorff_vals_in_plot.pkl', 'wb') as f:
# 	pickle.dump(hausdorff_vals_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

# open from pickle file
with open('data/hausdorff_vals_in_plot.pkl', 'rb') as f:
	hausdorff_vals_to_save = pickle.load(f)

hausdorff_dict_imu = hausdorff_vals_to_save['imu']
hausdorff_dict_vision = hausdorff_vals_to_save['vision']
hausdorff_dict_rhc = hausdorff_vals_to_save['rhc']
speeds = ['2_0', '2_5', '2_6', '2_7', '2_8', '2_9', "3_0", "3_1", "3_2"]
for key in hausdorff_dict_imu.keys():
	hausdorff_dict_imu[key] = [np.mean(hausdorff_dict_imu[key]), np.std(hausdorff_dict_imu[key])/4]
	hausdorff_dict_vision[key] = [np.mean(hausdorff_dict_vision[key]), np.std(hausdorff_dict_vision[key])/4]
	hausdorff_dict_rhc[key] = [np.mean(hausdorff_dict_rhc[key]), np.std(hausdorff_dict_rhc[key])/4]

# plot of vision, imu, rhc
plt.plot(np.arange(len(speeds)), [hausdorff_dict_vision[key][0] for key in speeds], '-.go')
plt.fill_between(np.arange(len(speeds)), [hausdorff_dict_vision[key][0] - hausdorff_dict_vision[key][1] for key in speeds], [hausdorff_dict_vision[key][0] + hausdorff_dict_vision[key][1] for key in speeds], alpha=0.2, color='green')

plt.plot(np.arange(len(speeds)), [hausdorff_dict_imu[key][0] for key in speeds], '-.mo')
plt.fill_between(np.arange(len(speeds)), [hausdorff_dict_imu[key][0] - hausdorff_dict_vision[key][1] for key in speeds], [hausdorff_dict_imu[key][0] + hausdorff_dict_vision[key][1] for key in speeds], alpha=0.2, color='magenta')

plt.plot(np.arange(len(speeds)), [hausdorff_dict_rhc[key][0] for key in speeds], '-.ro')
plt.fill_between(np.arange(len(speeds)), [hausdorff_dict_rhc[key][0] - hausdorff_dict_vision[key][1] for key in speeds], [hausdorff_dict_rhc[key][0] + hausdorff_dict_vision[key][1] for key in speeds], alpha=0.2, color='red')

# plt.ylim(0.1, 0.25)
# spacing = [0]
# for _ in range(len(speeds)-1):
# 	spacing += [spacing[-1] + 1.6]
# spacing = np.asarray(spacing)
#
# plt.bar(spacing, [hausdorff_dict_rhc[key][0] for key in speeds], yerr=[hausdorff_dict_rhc[key][1] for key in speeds], color='red', width=0.5)
# plt.bar(spacing+0.5, [hausdorff_dict_imu[key][0] for key in speeds], yerr=[hausdorff_dict_imu[key][1] for key in speeds], color='magenta', width=0.5)
# plt.bar(spacing+1.0, [hausdorff_dict_vision[key][0] for key in speeds], yerr=[hausdorff_dict_vision[key][1] for key in speeds], color='green', width=0.5)
# plt.show()
#
# plt.legend(['RHC', 'IMU', 'Vision'])
plt.xticks(np.arange(9), ['2.0', '2.5', '2.6', '2.7', '2.8', '2.9', '3.0', '3.1', '3.2'],)
plt.show()

# imu_data = np.asarray([hausdorff_dict_imu[key][:5] for key in speeds])
# vision_data = np.asarray([hausdorff_dict_vision[key][:5] for key in speeds])
#
# plt.violinplot(imu_data.T, np.arange(9))
# plt.violinplot(vision_data.T, np.arange(9)+0.5)
# plt.show()