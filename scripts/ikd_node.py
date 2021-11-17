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

        # image = message_filters.Subscriber("/terrain_patch/compressed", CompressedImage)
        image = message_filters.Subscriber("/webcam/image_raw/compressed", CompressedImage)

        odom = message_filters.Subscriber('/camera/odom/sample', Odometry)
        accel = message_filters.Subscriber('/camera/accel/sample', Imu)
        gyro = message_filters.Subscriber('/camera/gyro/sample', Imu)
        ts = message_filters.ApproximateTimeSynchronizer([image, odom, accel, gyro], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)

        self.data = {'image': [], 'odom': [], 'accel': [], 'gyro': []}
        self.n = 0

    def callback(self, image, odom, accel, gyro):
        self.n += 1
        print('Received messages :: ', self.n)

        # convert front cam image to top cam image
        bevimage = self.camera_imu_homography(odom, image)

        # convert odom to numpy array
        odom_np = np.array([odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])

        self.data['image'] = cv2.resize(bevimage, (128, 128), interpolation=cv2.INTER_AREA)
        self.data['accel'] = np.array([accel.linear_acceleration.x, accel.linear_acceleration.y, accel.linear_acceleration.z])
        self.data['gyro'] = np.array([gyro.angular_velocity.x, gyro.angular_velocity.y, gyro.angular_velocity.z])

        self.data['odom'].append(odom_np)
        self.data['odom'] = self.data['odom'][:self.history_len]
        

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

    def camera_imu_homography(self, odom, image):
        orientation_quat = [odom.pose.pose.orientation.x,
                            odom.pose.pose.orientation.y,
                            odom.pose.pose.orientation.z,
                            odom.pose.pose.orientation.w]
        # z_correction = odom.pose.pose.position.z

        C_i = np.array(
            [622.0649233612024, 0.0, 633.1717569157071, 0.0, 619.7990184421728, 368.0688607187958, 0.0, 0.0, 1.0]).reshape(
            (3, 3))

        R_imu_world = R.from_quat(orientation_quat)
        R_imu_world = R_imu_world.as_euler('xyz', degrees=True)
        R_imu_world[0], R_imu_world[1] = R_imu_world[0], -R_imu_world[1]
        R_imu_world[2] = 0.

        R_imu_world = R_imu_world
        R_imu_world = R.from_euler('xyz', R_imu_world, degrees=True)

        R_cam_imu = R.from_euler("xyz", [-90, 90, 0], degrees=True)
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
        output = output[420:520, 540:740]

        return output


class IKDNode(object):
  def __init__(self, model_path, config_path, history_len, input_topic, output_topic):
    self.model_path = model_path
    self.config_path = config_path
    self.history_len = history_len
    self.input_topic = input_topic
    self.output_topic = output_topic
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Loading Model...")
    self.model = VisualIKDNet(input_size=6 + 3*(self.history_len+1), output_size=2, hidden_size=256).to(device=self.device)
    if self.model_path is not None:
      self.model.load_state_dict(torch.load(self.model_path))
    print("Loaded Model")
    self.nav_cmd = AckermannCurvatureDriveMsg()
    self.nav_cmd.velocity = 0.0
    self.nav_cmd.curvature = 0.0
    
  def navCallback(self, msg):
    data = self.data_processor.get_data()
    odom_history = data['odom']
    desired_odom = [np.array([msg.velocity, 0, msg.velocity * msg.curvature])]
    accel = torch.tensor(data['accel'])
    gyro = torch.tensor(data['gyro'])
    patch = data['image']
    patch = torch.tensor(patch).permute(2, 0, 1).to(self.device)
    odom_input = np.concatenate((odom_history, desired_odom))
    odom_input = torch.tensor(odom_input.flatten())
    print("SHAPES", odom_input.shape, accel.shape, gyro.shape, patch.shape)
    non_visual_input = torch.cat((odom_input, accel, gyro)).to(self.device)

    output = self.model(non_visual_input.unsqueeze(0).float(), patch.unsqueeze(0).float())

    print("OUTPUT", output)
    v, w = output.squeeze(0).detach().cpu().numpy()

    print("VW", v, w)
    # populate with v and w
    self.nav_cmd.velocity = v
    self.nav_cmd.curvature = w / v
    self.nav_publisher.publish(self.nav_cmd)

  def listen(self):
    rospy.init_node('ikd_node', anonymous=True)

    self.data_processor = LiveDataProcessor(self.config_path, self.history_len)
    while (len(self.data_processor.get_data()['odom']) < self.history_len):
      print("Waiting for data processor initialization...Are all the necessary sensors running?")
      rospy.sleep(1)
    print("Data processor initialized, listening for commands")
    rospy.Subscriber(self.input_topic, AckermannCurvatureDriveMsg, self.navCallback)
    self.nav_publisher = rospy.Publisher(self.output_topic, AckermannCurvatureDriveMsg, queue_size=1)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='ikd node')
  parser.add_argument('--model_path', default='models/torchmodel.pt', type=str)
  parser.add_argument('--input_topic', default='/ackermann_drive_init', type=str)
  parser.add_argument('--output_topic', default='/ackermann_curvature_drive',  type=str)
  parser.add_argument('--config_path', type=str, default="config/alphatruck.yaml")
  parser.add_argument('--history_len', type=int, default=10)
  args = parser.parse_args()

  node = IKDNode(args.model_path, args.config_path, args.history_len, args.input_topic, args.output_topic)


  import signal

  def handler(signum, frame):
    print("Received signal {}, shutting down".format(signum))
    exit(signum)

  signal.signal(signal.SIGINT, handler)

  node.listen()




