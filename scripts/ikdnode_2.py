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

class IKDNode(object):
    def __init__(self, config_path, model_path, history_len, input_topic, output_topic):
        self.model_path = model_path
        self.config_path = config_path
        self.data = []
        self.history_len = history_len

        self.input_topic = input_topic
        self.output_topic = output_topic

        cprint('Reading Config file.. ', 'yellow')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            print(self.config)
        cprint('Parsed Config file successfully ', 'yellow', attrs=['blink'])

        # registering callback function for sensor data
        image = message_filters.Subscriber("/webcam/image_raw/compressed", CompressedImage)
        odom = message_filters.Subscriber('/camera/odom/sample', Odometry)
        accel = message_filters.Subscriber('/camera/accel/sample', Imu)
        gyro = message_filters.Subscriber('/camera/gyro/sample', Imu)
        vectornavimu = message_filters.Subscriber("/vectornav/IMU", Imu)
        ts = message_filters.ApproximateTimeSynchronizer([image, odom, accel, gyro, vectornavimu], 20, 0.05, allow_headerless=False)
        self.data = {'image': [], 'odom': [], 'accel': [], 'gyro': [], 'vectornavimu': []}

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

        rospy.Subscriber(self.input_topic, AckermannCurvatureDriveMsg, self.navCallback, queue_size=10)
        self.nav_publisher = rospy.Publisher(self.output_topic, AckermannCurvatureDriveMsg, queue_size=1)
        ts.registerCallback(self.callback)

    def navCallback(self, msg):
        print("Received Nav Command : ", msg.velocity, msg.curvature)

        if len(self.data['odom']) < self.history_len:
            print("Waiting for data processor initialization...Are all the necessary sensors running?")
            return

        odom_history = self.data['odom']
        desired_odom = [np.array([msg.velocity, 0, msg.velocity * msg.curvature])]
        accel = torch.tensor(self.data['accel'])
        gyro = torch.tensor(self.data['gyro'])

        patch = self.camera_imu_homography(self.data['vectornavimu'], self.data['image'])
        patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        patch = patch/255.0
        patch = torch.tensor(patch).permute(2, 0, 1).to(self.device)

        odom_input = np.concatenate((odom_history, desired_odom))
        odom_input = torch.tensor(odom_input.flatten())
        non_visual_input = torch.cat((odom_input, accel, gyro)).to(self.device)

        with torch.no_grad():
            output = self.model(non_visual_input.unsqueeze(0).float(), patch.unsqueeze(0).float())

        # print("desired : ", desired_odom)
        v, w = output.squeeze(0).detach().cpu().numpy()

        # populate with v and w
        self.nav_cmd.velocity = v
        self.nav_cmd.curvature = w / v
        print("Output Nav Command : ", v, w/v)

        self.nav_publisher.publish(self.nav_cmd)

    def callback(self, image, odom, accel, gyro, vectornavimu):
        # print('Received messages :: ', rospy.Time.now())

        # convert front cam image to top cam image
        # bevimage = self.camera_imu_homography(vectornavimu, image)
        # bevimage = cv2.resize(bevimage, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        # self.data['image'] = bevimage/255.0

        self.data['image'] = image
        self.data['vectornavimu'] = vectornavimu
        self.data['accel'] = np.array([accel.linear_acceleration.x, accel.linear_acceleration.y, accel.linear_acceleration.z])
        self.data['gyro'] = np.array([gyro.angular_velocity.x, gyro.angular_velocity.y, gyro.angular_velocity.z])

        # convert odom to numpy array
        odom_np = np.array([odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])
        self.data['odom'] = (self.data['odom'] + [odom_np])[:self.history_len]

    @staticmethod
    def homography_camera_displacement(R1, R2, t1, t2, n1):
        R12 = R2 @ R1.T
        t12 = R2 @ (- R1.T @ t1) + t2
        # d is distance from plane to t1.
        d = np.linalg.norm(n1.dot(t1.T))

        H12 = R12 - ((t12 @ n1.T) / d)
        H12 /= H12[2, 2]
        return H12

    def camera_imu_homography(self, imu, image):
        orientation_quat = [imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w]

        C_i = np.array(
            [622.0649233612024, 0.0, 633.1717569157071, 0.0, 619.7990184421728, 368.0688607187958, 0.0, 0.0,
             1.0]).reshape(
            (3, 3))

        R_imu_world = R.from_quat(orientation_quat)
        R_imu_world = R_imu_world.as_euler('xyz', degrees=True)
        R_imu_world[0], R_imu_world[1] = -R_imu_world[0], R_imu_world[1]
        R_imu_world[2] = 0.

        R_imu_world = R_imu_world
        R_imu_world = R.from_euler('xyz', R_imu_world, degrees=True)

        R_cam_imu = R.from_euler("xyz", [90, -90, 0], degrees=True)
        R1 = R_cam_imu * R_imu_world
        R1 = R1.as_matrix()

        R2 = R.from_euler("xyz", [0, 0, -90], degrees=True).as_matrix()
        t1 = R1 @ np.array([0., 0., 0.5]).reshape((3, 1))
        t2 = R2 @ np.array([-2.5, -0., 6.0]).reshape((3, 1))
        n = np.array([0, 0, 1]).reshape((3, 1))
        n1 = R1 @ n

        H12 = self.homography_camera_displacement(R1, R2, t1, t2, n1)
        homography_matrix = C_i @ H12 @ np.linalg.inv(C_i)
        homography_matrix /= homography_matrix[2, 2]

        img = np.fromstring(image.data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        output = cv2.warpPerspective(img, homography_matrix, (1280, 720))
        # output = output[420:520, 540:740]

        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ikd node')
    parser.add_argument('--model_path', default='models/torchmodel.pt', type=str)
    parser.add_argument('--input_topic', default='/ackermann_drive_init', type=str)
    parser.add_argument('--output_topic', default='/ackermann_curvature_drive',  type=str)
    parser.add_argument('--config_path', type=str, default="config/alphatruck.yaml")
    parser.add_argument('--history_len', type=int, default=10)
    args = parser.parse_args()

    rospy.init_node('ikd_node', anonymous=True)

    node = IKDNode(args.config_path, args.model_path, args.history_len, args.input_topic, args.output_topic)

    # import signal
    # def handler(signum, frame):
    #     print("Received signal {}, shutting down".format(signum))
    #     exit(signum)
    # signal.signal(signal.SIGINT, handler)
    # node.listen()

    # while not rospy.is_shutdown():
    rospy.spin()


