#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import glob
import rosbag
import yaml
print('yaml version :: ', yaml.__version__)

def read_expt_data_from_rosbags(rosbag_paths):
	data = {
		'graph_nav': {},
		'ikd_node': {},
	}

	# read every rosbag
	for rosbag_path in rosbag_paths:
		# open the rosbag
		bag = rosbag.Bag(rosbag_path)

		expt_type = None
		for key in data.keys():
			if key in rosbag_path:
				expt_type, expt_num = key, len(data[key]) + 1
				data[expt_type][expt_num] = []

		assert expt_type is not None, 'Could not find experiment type in rosbag path: {}'.format(rosbag_path)

		# read all messages in the rosbag
		for topic, msg, t in bag.read_messages(topics=['/localization']):
			# get the localization data
			data[expt_type][expt_num].append([msg.pose.x, msg.pose.y, msg.pose.theta])

	# convert everything to numpy arrays
	for expt_type in data.keys():
		for expt_num in data[expt_type].keys():
			data[expt_type][expt_num] = np.array(data[expt_type][expt_num])

	return data

def load_waypoints_position(waypoint_path):
	# get the ground truth waypoint data
	with open(waypoint_path) as f:
		waypoints = yaml.load(f, Loader=yaml.FullLoader)

	pos = []
	for key in waypoints.keys():
		pos.append(waypoints[key]['position'])
	pos = np.asarray(pos)
	return pos


if __name__ == '__main__':
	# get the rosbag paths
	rosbag_paths = glob.glob('data/expt_data/*.bag')
	# read the expt data from the rosbags
	data = read_expt_data_from_rosbags(rosbag_paths)

	waypoints = load_waypoints_position('data/expt_data/waypoints.yaml')


	for i in range(1, len(data['graph_nav'])+1):
		# plot the data
		len_pts = len(data['graph_nav'][1])
		plt.plot(data['graph_nav'][i][:, 0], data['graph_nav'][i][:, 1], 'ro-', label='Graph Navigation', markersize=1)
		plt.plot(data['ikd_node'][i][:, 0], data['ikd_node'][i][:, 1], 'bo-', label='IKD Node', markersize=1)
		plt.plot(waypoints[:, 0], waypoints[:, 1], 'ko-', label='Ground Truth', markersize=1)
		plt.legend()
		plt.show()
