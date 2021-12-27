#!/usr/env/python

import yaml
import rospy
from geometry_msgs.msg import PoseStamped
from termcolor import cprint
import roslib
roslib.load_manifest('amrl_msgs')
from amrl_msgs.msg import Localization2DMsg
from scipy.spatial.transform import Rotation as R

class WaypointRecorder:
	def __init__(self):
		self.waypoints = {}
		# self.waypoint_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.waypoint_callback)
		self.localization_sub = rospy.Subscriber('/localization', Localization2DMsg, self.localization_callback)

	def waypoint_callback(self, msg):
		cprint('Waypoint received', 'green', attrs=['bold'])
		num_wpts = len(self.waypoints.keys())

		self.waypoints[num_wpts+1] = {}
		self.waypoints[num_wpts+1]['position'] = [msg.pose.position.x, msg.pose.position.y]
		self.waypoints[num_wpts+1]['orientation'] = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]

		print('received waypoint : ', self.waypoints[num_wpts+1])

		print(msg.pose)

	def localization_callback(self, msg):
		cprint('Localization received', 'green', attrs=['bold'])
		num_wpts = len(self.waypoints.keys())
		self.waypoints[num_wpts+1] = {}
		self.waypoints[num_wpts+1]['position'] = [msg.pose.x, msg.pose.y]
		quaternion = R.from_euler('XYZ', [0, 0, msg.pose.theta], degrees=False).as_quat()
		self.waypoints[num_wpts+1]['orientation'] = [quaternion[0], quaternion[1], quaternion[2], quaternion[3]]
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