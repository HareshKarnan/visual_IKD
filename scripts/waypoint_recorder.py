#!/usr/env/python

import yaml
import rospy
from geometry_msgs.msg import PoseStamped
from termcolor import cprint

class WaypointRecorder:
	def __init__(self):
		self.waypoints = {}
		self.waypoint_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.waypoint_callback)

	def waypoint_callback(self, msg):
		cprint('Waypoint received', 'green', attrs=['bold'])
		num_wpts = len(self.waypoints.keys())

		self.waypoints[num_wpts+1] = {}
		self.waypoints[num_wpts+1]['xp'] = msg.pose.position.x
		self.waypoints[num_wpts+1]['yp'] = msg.pose.position.y
		self.waypoints[num_wpts+1]['zp'] = msg.pose.position.z
		self.waypoints[num_wpts+1]['xr'] = msg.pose.orientation.x
		self.waypoints[num_wpts+1]['yr'] = msg.pose.orientation.y
		self.waypoints[num_wpts+1]['zr'] = msg.pose.orientation.z
		self.waypoints[num_wpts+1]['wr'] = msg.pose.orientation.w

		print('received waypoint : ', self.waypoints[num_wpts+1])

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