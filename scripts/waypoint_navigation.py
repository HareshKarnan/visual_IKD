#!/usr/bin/env python
import rospy
import math
import json
from std_msgs.msg import String
import roslib

roslib.load_manifest('amrl_msgs')
import argparse
import time
import numpy as np
import yaml
from geometry_msgs.msg import PoseStamped
from amrl_msgs.msg import Localization2DMsg
from termcolor import cprint

parser = argparse.ArgumentParser()
parser.add_argument('--loop', action='store_true')
parser.add_argument('--waypoints', type=str, required=True, help='json file containing an array of waypoints')

args = parser.parse_args()

# read the yaml file and load
with open(args.waypoints) as f:
	waypoints = yaml.load(f)

print('waypoints : ', waypoints)

class WaypointNavigator():
	WAYPOINT_THRESHOLD = 0.75

	def __init__(self, waypoints):


		self.waypoints = waypoints

		self.current_waypoint = 1
		rospy.Subscriber("localization", Localization2DMsg, self.loc_callback)
		self.nav_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
		self.goal_msg = PoseStamped()
		self.goal_msg.header.frame_id = ""

		cprint('Initialized waypoint navigator', 'green', attrs=['bold'])

	def get_target_waypoint(self):
		if (self.current_waypoint >= len(self.waypoints)-10):
			if (args.loop):
				print("Circuit Complete, restarting...")
				# find the closest waypoint to the current location
				self.current_waypoint = self.get_closest_waypoint() + 10
				print('closest waypoint : ', self.current_waypoint)

				# self.current_waypoint = 1
			else:
				print("Completed waypoint navigation, exiting...")
				exit(0)

		return self.waypoints[self.current_waypoint]

	def get_closest_waypoint(self):
		closest_waypoint = 1
		closest_waypoint_dist = float('inf')
		for i in range(1, int(len(self.waypoints)/2)):
			waypoint = self.waypoints[i]
			dist = np.linalg.norm(np.array([self.loc.pose.x, self.loc.pose.y]) - np.array([waypoint["position"][0], waypoint["position"][1]]))
			if (dist < closest_waypoint_dist):
				closest_waypoint = i
				closest_waypoint_dist = dist

		return closest_waypoint

	def loc_callback(self, loc):
		self.loc = loc
		target_waypoint = self.get_target_waypoint()

		if WaypointNavigator.is_close(target_waypoint, loc.pose):
			self.current_waypoint = min(len(self.waypoints)-1, self.current_waypoint + 20)
			self.send_nav_command()

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
	def is_close(cls, target, pose):
		target_pos = target["position"]
		diff = np.linalg.norm(np.array([pose.x, pose.y]) - np.array([target_pos[0], target_pos[1]]))
		return diff < cls.WAYPOINT_THRESHOLD


def setup_ros_node():
	rospy.init_node('waypoint_navigation')

	waypoint_nav = WaypointNavigator(waypoints)
	time.sleep(1)
	waypoint_nav.send_nav_command()
	rospy.spin()


setup_ros_node()