#!/usr/bin/env python
import pickle

import matplotlib.pyplot as plt
import numpy as np
import rospy
import message_filters
import roslib
roslib.load_manifest('amrl_msgs')

from amrl_msgs.msg import Localization2DMsg
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker, MarkerArray

class TransformCompute():
	def __init__(self):
		localization_sub = message_filters.Subscriber('/localization', Localization2DMsg)
		realsense_sub = message_filters.Subscriber('/camera/odom/sample', Odometry)

		self.curr_pos_pub = rospy.Publisher('/curr_pos', Marker, queue_size=10)
		self.est_pos_pub = rospy.Publisher('/est_pos', Marker, queue_size=10)

		self.ts = message_filters.ApproximateTimeSynchronizer([localization_sub, realsense_sub], 10, 0.01, allow_headerless=True)
		self.ts.registerCallback(self.callback)
		self.rot, self.trans = [], []

	def callback(self, localization_msg, realsense_msg):

		loc_curr_pos = [localization_msg.pose.x, localization_msg.pose.y]
		self.publish_curr_pos_marker(loc_curr_pos)


		T_Emap_robot = self.get_affine_matrix_from_locomsg(localization_msg)
		T_Rmap_robot = self.get_affine_matrix_from_odommsg(realsense_msg)

		T_Emap_Rmap = T_Emap_robot @ np.linalg.pinv(T_Rmap_robot)

		rot = R.from_matrix(T_Emap_Rmap[:3, :3]).as_euler('xyz')[2]
		trans = T_Emap_Rmap[:3, 3]

		self.rot.append(rot)
		self.trans.append(trans)

		estRot = R.from_euler('xyz', [0, 0, np.mean(self.rot, axis=0)]).as_matrix()
		estTrans = np.mean(self.trans, axis=0).reshape((3, 1))
		estT_Emap_Rmap = np.vstack((np.hstack((estRot, estTrans)), np.asarray([0, 0, 0, 1])))

		estT_Emap_robot = estT_Emap_Rmap @ T_Rmap_robot

		self.publish_rs_est_pos_marker(estT_Emap_robot[:2, 3])



	@staticmethod
	def get_affine_matrix_from_locomsg(localization_msg):
		pos = np.asarray([localization_msg.pose.x, localization_msg.pose.y, 0.0]).reshape((3, 1))
		orient = R.from_euler('xyz', [0, 0, localization_msg.pose.theta]).as_matrix().reshape((3, 3))
		return np.vstack((np.hstack((orient, pos)), np.asarray([0, 0, 0, 1])))

	@staticmethod
	def get_affine_matrix_from_odommsg(realsense_msg):
		pos = np.asarray([realsense_msg.pose.pose.position.x, realsense_msg.pose.pose.position.y, 0.0]).reshape((3, 1))
		quat = [realsense_msg.pose.pose.orientation.x, realsense_msg.pose.pose.orientation.y, realsense_msg.pose.pose.orientation.z, realsense_msg.pose.pose.orientation.w]
		# z_axis = R.from_quat(quat).as_euler('xyz')[2]
		# orient = R.from_euler('xyz', [0, 0, z_axis]).as_matrix().reshape((3, 3))
		orient = R.from_quat(quat).as_matrix().reshape((3, 3))
		return np.vstack((np.hstack((orient, pos)), np.asarray([0, 0, 0, 1])))

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

	def compute_final_averaged_transform(self):
		rot = np.asarray(self.rot).mean(axis=0)
		rot_matrix = R.from_euler('xyz', [0, 0, rot]).as_matrix()
		trans = np.asarray(self.trans).mean(axis=0).reshape((3, 1))

		T_Emap_Rmap = np.vstack((np.hstack((rot_matrix, trans)), np.asarray([0, 0, 0, 1])))

		self.trans = np.asarray(self.trans)
		plt.figure()
		plt.subplot(3, 1, 1)
		plt.plot(self.trans[:, 0])
		plt.subplot(3, 1, 2)
		plt.plot(self.trans[:, 1])
		plt.subplot(3, 1, 3)
		plt.plot(self.rot)
		plt.show()

		# save recorded averages to pickle
		pickle.dump(T_Emap_Rmap, file=open('dump.pkl', 'wb'))


if __name__ == '__main__':
	rospy.init_node('realsense_map_transform')
	runner = TransformCompute()
	while not rospy.is_shutdown():
		rospy.spin()
	runner.compute_final_averaged_transform()