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

from scripts.model import SimpleIKDNet
import time
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

        # subscribe to accel and gyro topics
        rospy.Subscriber('/camera/accel/sample', Imu, self.accel_callback) #60 hz
        rospy.Subscriber('/camera/gyro/sample', Imu, self.gyro_callback) #60 hz

        ts = message_filters.ApproximateTimeSynchronizer([odom], 10, 0.05, allow_headerless=True)
        ts.registerCallback(self.callback)

        self.accel_msgs = np.zeros((60, 3), dtype=np.float32)
        self.gyro_msgs = np.zeros((200, 3), dtype=np.float32)

        self.data = {'accel': None, 'gyro': None, 'odom': []}
        self.n = 0

    def callback(self, odom):
        self.n += 1
        self.data['odom'].append(np.asarray([odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z]))

        # retain the history of odom
        self.data['odom'] = self.data['odom'][-self.history_len:]

        # retain the history of accel and gyro
        self.data['accel'] = self.accel_msgs.flatten()
        self.data['gyro'] = self.gyro_msgs.flatten()

    def get_data(self):
        return self.data

    def accel_callback(self, msg):
        # add to queue self.accel_msgs
        self.accel_msgs = np.roll(self.accel_msgs, -1, axis=0)
        self.accel_msgs[-1] = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

    def gyro_callback(self, msg):
        # add to queue self.gyro_msgs
        self.gyro_msgs = np.roll(self.gyro_msgs, -1, axis=0)
        self.gyro_msgs[-1] = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])


class IKDNode(object):
    def __init__(self, data_processor, model_path, history_len, input_topic, output_topic):
        self.model_path = model_path
        self.data_processor = data_processor

        self.history_len = history_len
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cprint('Using device: {}'.format(self.device), 'yellow', attrs=['bold'])

        print("Loading Model...")
        self.model = SimpleIKDNet(input_size=3*60 + 3*200 + 3 + 2,
                                  output_size=2,
                                  hidden_size=32).to(device=self.device)
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
        # if msg.velocity < 0.5:
        #     self.nav_cmd.velocity = msg.velocity
        #     self.nav_cmd.curvature = msg.curvature
        #     self.nav_publisher.publish(self.nav_cmd)
        #     return

        # if msg.velocity * msg.curvature > 1.0:
        #     cprint('Velocity * Curvature > 2.0', 'red', attrs=['bold'])
        #     self.nav_cmd.velocity = msg.velocity
        #     self.nav_cmd.curvature = msg.curvature
        #     self.nav_publisher.publish(self.nav_cmd)
        #     return


        data = self.data_processor.get_data()
        if len(data['odom']) < self.history_len:
            print("Waiting for data processor initialization...Are all the necessary sensors running?")
            return

        odom_history = np.asarray(data['odom']).flatten()
        desired_odom = np.array([msg.velocity, msg.velocity * msg.curvature])

        # form the input tensors
        accel = torch.tensor(data['accel']).to(device=self.device)
        gyro = torch.tensor(data['gyro']).to(device=self.device)
        odom_input = np.concatenate((odom_history, desired_odom))
        odom_input = torch.tensor(odom_input.flatten()).to(device=self.device)

        # non_visual_input = torch.cat((odom_input, accel, gyro)).to(self.device)
        start_time = time.time()
        with torch.no_grad():
            output = self.model(accel.unsqueeze(0).float(),
                                gyro.unsqueeze(0).float(),
                                odom_input.unsqueeze(0).float())

        # print("desired : ", desired_odom)
        v, w = output.squeeze(0).detach().cpu().numpy()
        print('time taken in seconds :: ', time.time() - start_time)

        print("Received Nav Command : ", msg.velocity, msg.velocity * msg.curvature)
        print("Output Nav Command : ", v, w)

        # populate with v and w
        self.nav_cmd.velocity = v
        self.nav_cmd.curvature = w/v

        self.nav_publisher.publish(self.nav_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ikd node')
    parser.add_argument('--model_path', default='models/torchmodel.pt', type=str)
    parser.add_argument('--input_topic', default='/ackermann_drive_init', type=str)
    parser.add_argument('--output_topic', default='/ackermann_curvature_drive',  type=str)
    parser.add_argument('--config_path', type=str, default="config/alphatruck.yaml")
    parser.add_argument('--history_len', type=int, default=1)
    args = parser.parse_args()

    rospy.init_node('ikd_node', anonymous=True)

    data_processor = LiveDataProcessor(args.config_path, args.history_len)
    node = IKDNode(data_processor, args.model_path, args.history_len, args.input_topic, args.output_topic)

    while not rospy.is_shutdown():
        rospy.spin()


