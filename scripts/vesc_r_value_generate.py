#!/usr/bin/env python3
"""
A simple utility function that listens to the wheel odom topic /odom and the
Intel Realsense odom topic /camera/odom/sample and computes the R value for
tuning the vesc driver.
For reference, see: https://docs.google.com/document/d/1KHScEcrj_tWRqCbMa7x2mVGDWvNHRvKcmOjuP4hD-ZQ/edit?usp=sharing
"""
import rospy
import numpy as np
from nav_msgs.msg import Odometry
import copy

class RValueFinder:
	def __init__(self):
		super(RValueFinder, self).__init__()
		self.distance_travelled_camera_odom, self.distance_travelled_wheel_odom = 0.0, 0.0
		self.camera_odom_prev_pos, self.wheel_odom_prev_pos = None, None
		self.initial_pos = None
		self.wheel_odom_msg = None

		rospy.Subscriber('/camera/odom/sample', Odometry, self.camera_odom_callback)
		rospy.Subscriber('/odom', Odometry, self.wheel_odom_callback)

		print('initialized Rvalue utility. Start driving the car now !')
		self.r_value_list = []

	def wheel_odom_callback(self, msg):
		self.wheel_odom_msg = msg

	def camera_odom_callback(self, camera_odom):
		wheel_odom = copy.deepcopy(self.wheel_odom_msg)

		if self.camera_odom_prev_pos is None or self.wheel_odom_prev_pos is None:
			self.camera_odom_prev_pos = [camera_odom.pose.pose.position.x, camera_odom.pose.pose.position.y]
			if self.wheel_odom_msg is None: return
			self.wheel_odom_prev_pos = [self.wheel_odom_msg.pose.pose.position.x, self.wheel_odom_msg.pose.pose.position.y]
			return

		self.distance_travelled_camera_odom += np.sqrt((camera_odom.pose.pose.position.x - self.camera_odom_prev_pos[0])**2 + (camera_odom.pose.pose.position.y - self.camera_odom_prev_pos[1])**2)
		self.distance_travelled_wheel_odom += np.sqrt((wheel_odom.pose.pose.position.x - self.wheel_odom_prev_pos[0])**2 + (wheel_odom.pose.pose.position.y - self.wheel_odom_prev_pos[1])**2)
		self.camera_odom_prev_pos = [camera_odom.pose.pose.position.x, camera_odom.pose.pose.position.y]
		self.wheel_odom_prev_pos = [wheel_odom.pose.pose.position.x, wheel_odom.pose.pose.position.y]

		# check if the distance travelled by camera odom is > 2m
		if self.distance_travelled_camera_odom > 2.0:
			r_val = self.distance_travelled_wheel_odom / self.distance_travelled_camera_odom
			self.r_value_list.append(r_val)
			print("R value : ", r_val, " # R value rolling mean: ", np.mean(self.r_value_list))
			self.distance_travelled_camera_odom, self.distance_travelled_wheel_odom = 0.0, 0.0
			self.initial_pos = None

if __name__ == '__main__':
	rospy.init_node('r_value_generator', anonymous=True)
	r_finder = RValueFinder()
	while not rospy.is_shutdown():
		rospy.spin()