#!/usr/bin/env python
import pickle

import rospy
import roslib
roslib.load_manifest('amrl_msgs')
import argparse
import time
import numpy as np
import yaml
from amrl_msgs.msg import Localization2DMsg
from nav_msgs.msg import Odometry
from termcolor import cprint
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import MarkerArray, Marker

def get_affine_from_odom(odom_msg):
	pos = np.asarray([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, 0]).reshape((3, 1))
	rot_mat = R.from_quat([odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w]).as_matrix()
	return np.vstack((np.hstack((rot_mat, pos)), np.array([0, 0, 0, 1])))

class WaypointNavigator():
	def __init__(self):

		# realsense odometry callback
		rospy.Subscriber('/camera/odom/sample', Odometry, self.callback)

		self.localization_spoof_topic = rospy.Publisher('/localization', Localization2DMsg, queue_size=10)

		cprint('Initialized Loco Spoofer', 'green', attrs=['bold'])

	def callback(self, realsense_msg):

		loco = Localization2DMsg()
		loco.header.frame_id = 'map'
		loco.header.stamp = rospy.Time.now()
		loco.pose.x = realsense_msg.pose.pose.position.x
		loco.pose.y = realsense_msg.pose.pose.position.y
		theta = R.from_quat([realsense_msg.pose.pose.orientation.x, realsense_msg.pose.pose.orientation.y, realsense_msg.pose.pose.orientation.z, realsense_msg.pose.pose.orientation.w]).as_euler('xyz')[2]
		loco.pose.theta = theta
		self.localization_spoof_topic.publish(loco)

if __name__ == '__main__':
	rospy.init_node('loco_spoofer')
	waypoint_nav = WaypointNavigator()
	time.sleep(1)

	# run at a rate
	runrate = rospy.Rate(20)

	while not rospy.is_shutdown():
		runrate.sleep()

	rospy.spin()

