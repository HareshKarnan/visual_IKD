#!/usr/bin/env python
import pickle

import matplotlib.pyplot as plt
import message_filters
import rospy
import math
import json
from std_msgs.msg import String
import roslib
import cv2

roslib.load_manifest('amrl_msgs')
import argparse
import time
import numpy as np
import yaml
from geometry_msgs.msg import PoseStamped
from amrl_msgs.msg import Localization2DMsg
from nav_msgs.msg import Odometry
from termcolor import cprint
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import MarkerArray, Marker

parser = argparse.ArgumentParser()
parser.add_argument('--no_loop', action='store_true')
parser.add_argument('--waypoints', type=str, default='waypoints.yaml')
parser.add_argument('--resample_number', type=int, default=25)
args = parser.parse_args()

enml_real_affine = pickle.load(open('dump.pkl', 'rb'))

def load_waypoints(waypoint_path):
	# get the ground truth waypoint data
	with open(waypoint_path) as f:
		waypoints = yaml.load(f)

	pos = []
	for key in waypoints.keys():
		pos.append(waypoints[key]['position'])
	pos = np.asarray(pos)
	return pos

# read the yaml file and load
with open(args.waypoints) as f:
	waypoints = yaml.load(f, Loader=yaml.FullLoader)

# print('waypoints : ', waypoints)

def resample_waypoints(waypoints, factor):
	resample_waypoints = {}
	for i, key in enumerate(list(waypoints.keys())[::factor]):
		resample_waypoints[i] = waypoints[key]
	return resample_waypoints

def smoothen_waypoints(waypoints):
	pos_arr = []
	for key in waypoints.keys():
		pos_arr.append(waypoints[key]['position'])

	pos_arr = np.asarray(pos_arr)

	print('pos arr shape : ', pos_arr.shape)
	pos_arr[:, 0] = savgol_filter(pos_arr[:, 0], 99, 3)
	pos_arr[:, 1] = savgol_filter(pos_arr[:, 1], 99, 3)
	for i, key in enumerate(waypoints.keys()):
		waypoints[key]['position'] = pos_arr[i]
	return waypoints, pos_arr

class WaypointNavigator():
	WAYPOINT_THRESHOLD = 1.75
	def __init__(self, waypoints, visualize=False, resample_num=5):

		self.waypoints = waypoints
		self.waypoints, self.pos_arr = smoothen_waypoints(self.waypoints)
		self.waypoints = resample_waypoints(waypoints, resample_num)
		# print(self.waypoints)

		# visualize the waypoints
		self.waypoint_pub = rospy.Publisher('/waypoints', MarkerArray, queue_size=10)
		self.curr_pos_pub = rospy.Publisher('/curr_pos', Marker, queue_size=10)
		self.est_pos_pub = rospy.Publisher('/est_pos', Marker, queue_size=10)
		self.next_pos_pub = rospy.Publisher('/next_pos', Marker, queue_size=10)

		print('total waypoints :: ', len(self.waypoints))
		self.visualize = visualize

		self.current_waypoint = 1
		self.counter = 0
		# rospy.Subscriber("localization", Localization2DMsg, self.loc_callback)
		# rospy.Subscriber('/camera/odom/sample', Odometry, self.realsense_callback)
		self.fig = plt.figure()
		self.fig.canvas.draw()

		realsense = message_filters.Subscriber('/camera/odom/sample', Odometry)
		localization = message_filters.Subscriber('localization', Localization2DMsg)
		ts = message_filters.ApproximateTimeSynchronizer([realsense, localization], 20, 0.2, allow_headerless=True)
		ts.registerCallback(self.realsense_callback)

		self.nav_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
		self.goal_msg = PoseStamped()
		self.goal_msg.header.frame_id = ""

		cprint('Initialized waypoint navigator', 'green', attrs=['bold'])


	def get_target_waypoint(self):
		if (self.current_waypoint > len(self.waypoints)-1):
			if (not args.no_loop):
				print("Circuit Complete, restarting...")
				# find the closest waypoint to the current location
				self.current_waypoint = self.get_closest_waypoint()
				print('closest waypoint : ', self.current_waypoint)
			else:
				print("Completed waypoint navigation, exiting...")
				exit(0)

		return self.waypoints[self.current_waypoint]

	def get_closest_waypoint(self):
		closest_waypoint = 1
		closest_waypoint_dist = float('inf')
		for i in range(1, 5):
			waypoint = self.waypoints[i]
			dist = np.linalg.norm(np.array([self.loc.pose.x, self.loc.pose.y]) - np.array([waypoint["position"][0], waypoint["position"][1]]))
			if (dist < closest_waypoint_dist):
				closest_waypoint = i
				closest_waypoint_dist = dist
		cprint('Found closest waypoint as :: ' + str(closest_waypoint), 'green', attrs=['bold'])
		return closest_waypoint

	def realsense_callback(self, realsense_msg, localization_msg):
		self.loc = localization_msg

		# get the odometry in map frame
		realsense_curr_pos = self.get_pos_in_map_frame(realsense_msg)
		self.publish_rs_est_pos_marker(realsense_curr_pos)

		curr_position = [localization_msg.pose.x, localization_msg.pose.y]
		self.publish_curr_pos_marker(curr_position)

		target_waypoint = self.get_target_waypoint()
		target_position = [target_waypoint["position"][0], target_waypoint["position"][1]]
		self.publish_next_pos_marker(target_position)

		if WaypointNavigator.is_close(target_waypoint, curr_position):
			self.current_waypoint = min(len(self.waypoints), self.current_waypoint + 1)
			self.send_nav_command()

	def publish_next_pos_marker(self, position):
		marker = Marker()
		marker.header.frame_id = "camera_odom_frame"
		marker.header.stamp = rospy.Time.now()
		marker.ns = "next_pos"
		marker.id = 0
		marker.type = Marker.CUBE
		marker.action = Marker.ADD
		marker.pose.position.x = position[0]
		marker.pose.position.y = position[1]
		marker.pose.position.z = 0
		marker.pose.orientation.x = 0.0
		marker.pose.orientation.y = 0.0
		marker.pose.orientation.z = 0.0
		marker.pose.orientation.w = 1.0
		marker.color.r = 0
		marker.color.g = 0
		marker.color.b = 1
		marker.color.a = 1.0
		marker.scale.x = 0.2
		marker.scale.y = 0.2
		marker.scale.z = 0.2
		self.next_pos_pub.publish(marker)

	def publish_rs_est_pos_marker(self, position):
		marker = Marker()
		marker.header.frame_id = "camera_odom_frame"
		marker.header.stamp = rospy.Time.now()
		marker.ns = "est_pos"
		marker.id = 0
		marker.type = Marker.CUBE
		marker.action = Marker.ADD
		marker.pose.position.x = position[0]
		marker.pose.position.y = position[1]
		marker.pose.position.z = 0
		marker.pose.orientation.x = 0.0
		marker.pose.orientation.y = 0.0
		marker.pose.orientation.z = 0.0
		marker.pose.orientation.w = 1.0
		marker.color.r = 0
		marker.color.g = 1
		marker.color.b = 0
		marker.color.a = 1.0
		marker.scale.x = 0.2
		marker.scale.y = 0.2
		marker.scale.z = 0.2
		self.est_pos_pub.publish(marker)

	def publish_curr_pos_marker(self, curr_position):
		marker = Marker()
		marker.header.frame_id = "camera_odom_frame"
		marker.header.stamp = rospy.Time.now()
		marker.ns = "curr_pos"
		marker.id = 0
		marker.type = Marker.CUBE
		marker.action = Marker.ADD
		marker.pose.position.x = curr_position[0]
		marker.pose.position.y = curr_position[1]
		marker.pose.position.z = 0
		marker.pose.orientation.x = 0.0
		marker.pose.orientation.y = 0.0
		marker.pose.orientation.z = 0.0
		marker.pose.orientation.w = 1.0
		marker.color.r = 1
		marker.color.g = 0
		marker.color.b = 0
		marker.color.a = 1.0
		marker.scale.x = 0.2
		marker.scale.y = 0.2
		marker.scale.z = 0.2
		self.curr_pos_pub.publish(marker)

	def visualize_waypoints(self):
		markerarray = MarkerArray()
		for i, key in enumerate(self.waypoints.keys()):
			marker = Marker()
			marker.header.frame_id = "camera_odom_frame"
			marker.header.stamp = rospy.Time.now()
			marker.ns = "waypoints"
			marker.id = i
			marker.type = marker.CUBE
			marker.action = marker.ADD
			marker.pose.position.x = self.waypoints[key]['position'][0]
			marker.pose.position.y = self.waypoints[key]['position'][1]
			# marker.pose.position.z = 0
			marker.pose.orientation.x = 0.0
			marker.pose.orientation.y = 0.0
			marker.pose.orientation.z = 0.0
			marker.pose.orientation.w = 1.0
			marker.color.a = 1.0
			marker.color.r = 0.0
			marker.color.g = 1.0
			marker.color.b = 0.0
			marker.scale.x = 0.1
			marker.scale.y = 0.1
			marker.scale.z = 0.1
			markerarray.markers.append(marker)
		self.waypoint_pub.publish(markerarray)

	@staticmethod
	def get_pos_in_map_frame(realsense_msg):
		pos = np.asarray([realsense_msg.pose.pose.position.x, realsense_msg.pose.pose.position.y]).reshape((2, 1))
		quat = [realsense_msg.pose.pose.orientation.x, realsense_msg.pose.pose.orientation.y, realsense_msg.pose.pose.orientation.z, realsense_msg.pose.pose.orientation.w]
		# orient = R.from_quat(quat).as_matrix().reshape((3, 3))
		z_axis = R.from_quat(quat).as_euler('xyz')[2]
		orient = R.from_euler('xyz', [0, 0, z_axis]).as_matrix()#.reshape((3, 3))
		realsense_affine = np.vstack((np.hstack((orient[:2, :2], pos)), np.asarray([0, 0, 1])))

		enml_robot = np.matmul(enml_real_affine, realsense_affine)
		return enml_robot[:2, 2].flatten()


	def send_nav_command(self):
		target_waypoint = self.get_target_waypoint()
		print("Navigating to ... \n", self.current_waypoint, '/', len(self.waypoints))

		self.goal_msg.pose.position.x = target_waypoint["position"][0]
		self.goal_msg.pose.position.y = target_waypoint["position"][1]
		self.goal_msg.pose.orientation.x = target_waypoint["orientation"][0]
		self.goal_msg.pose.orientation.y = target_waypoint["orientation"][1]
		self.goal_msg.pose.orientation.z = target_waypoint["orientation"][2]
		self.goal_msg.pose.orientation.w = target_waypoint["orientation"][3]

		self.nav_pub.publish(self.goal_msg)

	@classmethod
	def is_close(cls, target, curr_position):
		target_pos = target["position"]
		diff = np.linalg.norm(np.array([curr_position[0], curr_position[1]]) - np.array([target_pos[0], target_pos[1]]))
		return diff < cls.WAYPOINT_THRESHOLD

if __name__ == '__main__':
	rospy.init_node('waypoint_navigation')
	waypoint_nav = WaypointNavigator(waypoints, resample_num=args.resample_number)
	time.sleep(1)

	# run at a rate
	runrate = rospy.Rate(10)

	while not rospy.is_shutdown():
		waypoint_nav.send_nav_command()
		waypoint_nav.visualize_waypoints()
		runrate.sleep()

	rospy.spin()

