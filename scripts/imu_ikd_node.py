#!/usr/bin/python3
import argparse
import rospy
import torch
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage, Imu
from nav_msgs.msg import Odometry

import message_filters
from termcolor import cprint
import yaml
from scipy.spatial.transform import Rotation as R

from scripts.model import VisualIKDNet

import roslib
roslib.load_manifest('amrl_msgs')
from amrl_msgs.msg import AckermannCurvatureDriveMsg

class LiveDataProcessor(object):
    def __init__(self, config_path, history_len):
        self.data = []
        self.config_path = config_path
        self.history_len = history_len

        with open(config_path, 'r') as f:
            cprint('Reading Config file.. ', 'yellow')
            self.config = yaml.safe_load(f)
            cprint('Parsed Config file successfully ', 'yellow', attrs=['blink'])
            print(self.config)

        odom = message_filters.Subscriber('/camera/odom/sample', Odometry)
        accel = message_filters.Subscriber('/camera/accel/sample', Imu)
        gyro = message_filters.Subscriber('/camera/gyro/sample', Imu)

        ts = message_filters.ApproximateTimeSynchronizer([odom, accel, gyro], 20, 0.05, allow_headerless=True)
        ts.registerCallback(self.callback)

        self.data = {'accel': [], 'gyro': [], 'odom': []}
        self.n = 0

    def callback(self, odom, accel, gyro):
        self.n += 1
        print('Received messages :: ', self.n)


        self.data['odom'].append(np.array([odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z]))
        self.data['accel'].append(np.array([accel.linear_acceleration.x, accel.linear_acceleration.y, accel.linear_acceleration.z]))
        self.data['gyro'].append(np.array([gyro.angular_velocity.x, gyro.angular_velocity.y, gyro.angular_velocity.z]))

        # retain the history
        self.data = {k: v[-self.history_len:] for k, v in self.data.items()}

    def get_data(self):
        return self.data

    @staticmethod
    def homography_camera_displacement(R1, R2, t1, t2, n1):
        R12 = R2 @ R1.T
        t12 = R2 @ (- R1.T @ t1) + t2
        # d is distance from plane to t1.
        d = np.linalg.norm(n1.dot(t1.T))

        H12 = R12 - ((t12 @ n1.T) / d)
        H12 /= H12[2, 2]
        return H12


class IKDNode(object):
    def __init__(self, data_processor, model_path, history_len, input_topic, output_topic):
        self.model_path = model_path
        self.data_processor = data_processor

        self.history_len = history_len
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Loading Model...")
        self.model = VisualIKDNet(input_size=6 + 3*(self.history_len+1), output_size=2, hidden_size=32).to(device=self.device)
        if self.model_path is not None:
            self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        print("Loaded Model")
        self.nav_cmd = AckermannCurvatureDriveMsg()
        self.nav_cmd.velocity = 0.0
        self.nav_cmd.curvature = 0.0

        rospy.Subscriber(self.input_topic, AckermannCurvatureDriveMsg, self.navCallback)
        self.nav_publisher = rospy.Publisher(self.output_topic, AckermannCurvatureDriveMsg, queue_size=1)

    def navCallback(self, msg):
        print("Received Nav Command : ", msg.velocity, msg.curvature)

        data = self.data_processor.get_data()
        if len(data['odom']) < self.history_len:
            print("Waiting for data processor initialization...Are all the necessary sensors running?")
            return
        else:
            print("Data processor initialized, listening for commands")

        odom_history = data['odom']
        desired_odom = [np.array([msg.velocity, 0, msg.velocity * msg.curvature])]

        # form the input tensors
        accel = torch.tensor(data['accel'])
        gyro = torch.tensor(data['gyro'])
        odom_input = np.concatenate((odom_history, desired_odom))
        odom_input = torch.tensor(odom_input.flatten())

        non_visual_input = torch.cat((odom_input, accel, gyro)).to(self.device)

        with torch.no_grad():
            output = self.model(non_visual_input.unsqueeze(0).float())

        # print("desired : ", desired_odom)
        v, w = output.squeeze(0).detach().cpu().numpy()

        # populate with v and w
        self.nav_cmd.velocity = v
        self.nav_cmd.curvature = w / v
        print("Output Nav Command : ", v, w/v)

        self.nav_publisher.publish(self.nav_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ikd node')
    parser.add_argument('--model_path', default='models/torchmodel.pt', type=str)
    parser.add_argument('--input_topic', default='/ackermann_drive_init', type=str)
    parser.add_argument('--output_topic', default='/ackermann_curvature_drive',  type=str)
    parser.add_argument('--config_path', type=str, default="config/alphatruck.yaml")
    parser.add_argument('--history_len', type=int, default=10)
    args = parser.parse_args()

    rospy.init_node('ikd_node', anonymous=True)

    data_processor = LiveDataProcessor(args.config_path, args.history_len)
    node = IKDNode(data_processor, args.model_path, args.history_len, args.input_topic, args.output_topic)

    while not rospy.is_shutdown():
        rospy.spin()


