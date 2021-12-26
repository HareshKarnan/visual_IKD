#!/usr/env/python

import yaml
import rospy
from geometry_msgs.msg import PoseStamped

class WaypointRecorder:
	def __init__(self):
		self.waypoints = []
		self.waypoint_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.waypoint_callback)

	def waypoint_callback(self, msg):
		self.waypoints.append(msg.pose)

if __name__ == '__main__':
	rospy.init_node('waypoint_recorder')
	waypoint_recorder = WaypointRecorder()
	while not rospy.is_shutdown():
		rospy.spin()

	with open('waypoints.yaml', 'w') as f:
		yaml.dump(waypoint_recorder.waypoints, f)

	print('Saved waypoints to waypoints.yaml')
	exit(0)