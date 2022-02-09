#!/usr/env/python

import yaml
import rospy
from geometry_msgs.msg import PoseStamped
from termcolor import cprint
import roslib
roslib.load_manifest('amrl_msgs')
# from amrl_msgs.msg import Localization2DMsg
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

class WaypointRecorder:
	def __init__(self):
		self.waypoints = {}
		self.localization_sub = rospy.Subscriber('/camera/odom/sample', Odometry, self.localization_callback)

	def waypoint_callback(self, msg):
		cprint('Waypoint received', 'green', attrs=['bold'])
		num_wpts = len(self.waypoints.keys())

		self.waypoints[num_wpts+1] = {}
		self.waypoints[num_wpts+1]['position'] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
		self.waypoints[num_wpts+1]['orientation'] = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]

		print('received waypoint : ', self.waypoints[num_wpts+1])

		print(msg.pose)

	def localization_callback(self, msg):
		cprint('Localization received', 'green', attrs=['bold'])
		num_wpts = len(self.waypoints.keys())
		self.waypoints[num_wpts+1] = {}
		self.waypoints[num_wpts+1]['position'] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
		self.waypoints[num_wpts+1]['orientation'] = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
		print(msg.pose)

if __name__ == '__main__':
	rospy.init_node('waypoint_recorder')
	waypoint_recorder = WaypointRecorder()
	while not rospy.is_shutdown():
		rospy.spin()

	with open('waypoints.yaml', 'w') as f:
		yaml.dump(waypoint_recorder.waypoints, f)

	print('Saved waypoints to waypoints.yaml')
	exit(0)