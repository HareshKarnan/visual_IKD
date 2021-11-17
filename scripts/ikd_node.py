
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

from geometry_msgs.msg import Twist

from scripts.train import IKDModel

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

        self.data['image'] =  bevimage
        self.data['accel'] = accel
        self.data['gyro'] = gyro

        self.data['odom'].append(odom_np)
        if (self.n > self.history_len):
            self.data['odom'].pop(0)

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

    @staticmethod
    def process_accel_gyro_data(data):
        for i in range(len(data['accel'])):
            accel = data['accel'][i].linear_acceleration
            gyro = data['gyro'][i].angular_velocity
            data['accel'][i] = [accel.x, accel.y, accel.z]
            data['gyro'][i] = [gyro.x, gyro.y, gyro.z]
        data['accel'] = np.asarray(data['accel'])
        data['gyro'] = np.asarray(data['gyro'])
        return data


class IKDNode(object):
  def __init__(self, model_path, config_path, history_len):
    self.model_path = model_path
    self.config_path = config_path
    self.history_len = history_len
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.model = IKDModel(input_size=6 + 3*self.history_len, output_size=2*self.history_len, hidden_size=256).to(device=self.device)
    if self.model_path is not None:
      self.model.load_state_dict(torch.load(self.model_path)["state_dict"])
    self.nav_cmd = Twist()
    self.nav_cmd.linear.x = 0.
    self.nav_cmd.linear.y = 0.
    self.nav_cmd.linear.z = 0.
    self.nav_cmd.angular.x = 0.
    self.nav_cmd.angular.y = 0.
    self.nav_cmd.angular.z = 0.
    
  def navCallback(self, msg):
    data = self.data_processor.get_data()
    odom_history = data['odom']
    desired_odom = [np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z])]
    accel = data['accel']
    gyro = data['gyro']
    patch = data['image']
    patch = torch.tensor(patch).permute(0, 3, 1, 2).to(self.device)
    odom_input = odom_history.concat(desired_odom)
    non_visual_input = torch.cat((np.array(odom_input).flatten(), accel, gyro), dim=1).to(self.device)

    output = self.model(non_visual_input.float(), patch.float())

    v, w = output.detach().cpu().numpy()

    # populate with v and w
    self.nav_cmd.linear.x = v
    self.nav_cmd.angular.z = w
    self.nav_publisher.publish(self.nav_cmd)

  def listen(self):
    rospy.init_node('ikd_node', anonymous=True)

    self.data_processor = LiveDataProcessor(self.config_path, self.history_len)
    while (self.data_processor.n < 10):
      print("Waiting for data processor initialization...")
      rospy.sleep(1)
    print("Data processor initialized, listening for commands")
    rospy.Subscriber("navigation/cmd_vel_init", Twist, self.navCallback)
    self.nav_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='ikd node')
  parser.add_argument('--model_path',  type=str)
  parser.add_argument('--config_path', type=str, default="config/alphatruck.yaml")
  parser.add_argument('--history_len', type=int, default=20)
  args = parser.parse_args()

  node = IKDNode(args.model_path, args.config_path, args.history_len)

  import signal

  def handler(signum, frame):
    print("Received signal {}, shutting down".format(signum))
    exit(signum)

  signal.signal(signal.SIGINT, handler)

  node.listen()




