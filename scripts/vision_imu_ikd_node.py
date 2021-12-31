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

PATCH_SIZE = 64
PATCH_EPSILON = 0.2 * PATCH_SIZE * PATCH_SIZE
ACTUATION_LATENCY = 0.1
C_i = np.array(
    [622.0649233612024, 0.0, 633.1717569157071, 0.0, 619.7990184421728, 368.0688607187958, 0.0, 0.0, 1.0]).reshape(
    (3, 3))

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
        image = message_filters.Subscriber("/webcam/image_raw/compressed", CompressedImage)
        vectornavimu = message_filters.Subscriber("/vectornav/IMU", Imu)

        # subscribe to accel and gyro topics
        rospy.Subscriber('/camera/accel/sample', Imu, self.accel_callback) #60 hz
        rospy.Subscriber('/camera/gyro/sample', Imu, self.gyro_callback) #200 hz

        ts = message_filters.ApproximateTimeSynchronizer([odom, image, vectornavimu], 10, 0.05, allow_headerless=True)
        ts.registerCallback(self.callback)

        self.accel_msgs = np.zeros((60, 3), dtype=np.float32)
        self.gyro_msgs = np.zeros((200, 3), dtype=np.float32)

        self.data = {'accel': None, 'gyro': None, 'odom': None, 'patch': None}
        self.history_storage = {'bevimage': [], 'odom_msg': []}
        self.data_ready = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def callback(self, odom, image, vectornavimu):
        # populate the data dictionary
        self.data['odom'] = np.array([odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])
        self.data['accel'] = torch.tensor(self.accel_msgs.flatten()).to(self.device).unsqueeze(0).float()
        self.data['gyro'] = torch.tensor(self.gyro_msgs.flatten()).to(self.device).unsqueeze(0).float()

        # get the bird's eye view image
        bevimage = self.camera_imu_homography(vectornavimu, image)

        # add this to the trailing history
        self.history_storage['bevimage'] = self.history_storage['bevimage'][-4:] + [bevimage]
        self.history_storage['odom_msg'] = self.history_storage['odom_msg'][-4:] + [odom]

        # check if 5 frames have been collected
        if len(self.history_storage['bevimage']) < 5:
            cprint('Not enough frames. Waiting for more frames to accumulate')
            return

        # if code reaches here, then 5 frames have been collected and we are ready to serve data for the model
        self.data_ready = True

        # search for the patch in the past 5 frames
        # found_patch, patch = False, None
        # for j in range(5, -1, -1):
        #     prev_image = self.history_storage['bevimage'][j]
        #     prev_odom = self.history_storage['odom_msg'][j]
        #     patch, patch_black_pct, curr_img, vis_img = self.get_patch_from_odom_delta(
        #         odom.pose.pose, prev_odom.pose.pose, odom.twist.twist,
        #         prev_odom.twist.twist, prev_image, bevimage)
        #     if patch is not None:
        #         # patch has been found. Stop searching
        #         cprint('Found patch in the past 5 frames', 'green', attrs=['bold'])
        #         found_patch = True
        #         break
        #
        # if not found_patch:
        #     cprint('Could not find patch in the past 5 frames', 'red', attrs=['bold'])
        patch = bevimage[500:564, 613:677]
        patch = patch.astype(np.float32)
        patch = patch / 255.0
        patch = torch.tensor(patch).unsqueeze(0).to(self.device).float()
        self.data['patch'] = patch.permute(0, 3, 1, 2)

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

    @staticmethod
    def camera_imu_homography(imu, image):
        orientation_quat = [imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w]

        R_imu_world = R.from_quat(orientation_quat)
        R_imu_world = R_imu_world.as_euler('xyz', degrees=True)
        R_imu_world[0], R_imu_world[1], R_imu_world[2] = -R_imu_world[0], R_imu_world[1], 0.

        R_imu_world = R.from_euler('xyz', R_imu_world, degrees=True)
        R_cam_imu = R.from_euler("xyz", [90, -90, 0], degrees=True)
        R1 = R_cam_imu * R_imu_world
        R1 = R1.as_matrix()

        R2 = R.from_euler("xyz", [0, 0, -90], degrees=True).as_matrix()
        t1 = R1 @ np.array([0., 0., 0.5]).reshape((3, 1))
        t2 = R2 @ np.array([-2.5, -0., 6.0]).reshape((3, 1))
        n1 = R1 @ np.array([0, 0, 1]).reshape((3, 1))

        H12 = LiveDataProcessor.homography_camera_displacement(R1, R2, t1, t2, n1)
        homography_matrix = C_i @ H12 @ np.linalg.inv(C_i)
        homography_matrix /= homography_matrix[2, 2]

        img = np.fromstring(image.data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        output = cv2.warpPerspective(img, homography_matrix, (1280, 720))
        # flip output horizontally
        output = cv2.flip(output, 1)

        return output

    @staticmethod
    def homography_camera_displacement(R1, R2, t1, t2, n1):
        R12 = R2 @ R1.T
        t12 = R2 @ (- R1.T @ t1) + t2
        # d is distance from plane to t1.
        d = np.linalg.norm(n1.dot(t1.T))

        H12 = R12 - ((t12 @ n1.T) / d)
        H12 /= H12[2, 2]
        return H12

    @staticmethod
    def get_patch_from_odom_delta(curr_pos, prev_pos, curr_vel, prev_vel, prev_image, curr_image):
        curr_pos_np = np.array([curr_pos.position.x, curr_pos.position.y, 1])
        prev_pos_transform = np.zeros((3, 3))
        z_angle = R.from_quat(
            [prev_pos.orientation.x, prev_pos.orientation.y, prev_pos.orientation.z, prev_pos.orientation.w]).as_euler(
            'xyz', degrees=False)[2]
        prev_pos_transform[:2, :2] = R.from_euler('xyz', [0, 0, z_angle]).as_matrix()[:2, :2]  # figure this out
        prev_pos_transform[:, 2] = np.array([prev_pos.position.x, prev_pos.position.y, 1]).reshape((3))

        inv_pos_transform = np.linalg.inv(prev_pos_transform)
        curr_z_angle = R.from_quat(
            [curr_pos.orientation.x, curr_pos.orientation.y, curr_pos.orientation.z, curr_pos.orientation.w]).as_euler(
            'xyz', degrees=False)[2]
        curr_z_rotation = R.from_euler('xyz', [0, 0, curr_z_angle]).as_matrix()
        projected_loc_np = curr_pos_np + ACTUATION_LATENCY * (
                    curr_z_rotation @ np.array([curr_vel.linear.x, curr_vel.linear.y, 0]))

        patch_corners = [
            projected_loc_np + curr_z_rotation @ np.array([0.3, 0.3, 0]),
            projected_loc_np + curr_z_rotation @ np.array([0.3, -0.3, 0]),
            projected_loc_np + curr_z_rotation @ np.array([-0.3, -0.3, 0]),
            projected_loc_np + curr_z_rotation @ np.array([-0.3, 0.3, 0])
        ]
        patch_corners_prev_frame = [
            inv_pos_transform @ patch_corners[0],
            inv_pos_transform @ patch_corners[1],
            inv_pos_transform @ patch_corners[2],
            inv_pos_transform @ patch_corners[3],
        ]
        scaled_patch_corners = [
            (patch_corners_prev_frame[0] * 200).astype(np.int),
            (patch_corners_prev_frame[1] * 200).astype(np.int),
            (patch_corners_prev_frame[2] * 200).astype(np.int),
            (patch_corners_prev_frame[3] * 200).astype(np.int),
        ]

        CENTER = np.array((720, 640))
        patch_corners_image_frame = [
            CENTER + np.array((-scaled_patch_corners[0][1], -scaled_patch_corners[0][0])),
            CENTER + np.array((-scaled_patch_corners[1][1], -scaled_patch_corners[1][0])),
            CENTER + np.array((-scaled_patch_corners[2][1], -scaled_patch_corners[2][0])),
            CENTER + np.array((-scaled_patch_corners[3][1], -scaled_patch_corners[3][0]))
        ]
        vis_img = prev_image.copy()

        projected_loc_prev_frame = inv_pos_transform @ projected_loc_np
        scaled_projected_loc = (projected_loc_prev_frame * 200).astype(np.int)
        projected_loc_image_frame = CENTER + np.array((-scaled_projected_loc[1], -scaled_projected_loc[0]))
        cv2.circle(vis_img, (projected_loc_image_frame[0], projected_loc_image_frame[1]), 3, (0, 255, 255))

        persp = cv2.getPerspectiveTransform(np.float32(patch_corners_image_frame),
                                            np.float32([[0, 0], [63, 0], [63, 63], [0, 63]]))

        patch = cv2.warpPerspective(
            prev_image,
            persp,
            (64, 64)
        )

        zero_count = np.logical_and(np.logical_and(patch[:, :, 0] == 0, patch[:, :, 1] == 0), patch[:, :, 2] == 0)

        if np.sum(zero_count) > PATCH_EPSILON:
            return None, 1.0, None, None

        return patch, (np.sum(zero_count) / (64. * 64.)), curr_image, vis_img

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
        self.model = VisualIKDNet(input_size=3*60 + 3*200 + 3 + 2,
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
        data = self.data_processor.get_data()
        if not self.data_processor.data_ready:
            print("Waiting for data processor initialization...Are all the necessary sensors running?")
            return
        accel, gyro, patch = data['accel'], data['gyro'], data['patch']

        odom_history = np.asarray(data['odom']).flatten()
        desired_odom = np.array([msg.velocity, msg.velocity * msg.curvature])
        odom_input = np.concatenate((odom_history, desired_odom))
        odom_input = torch.tensor(odom_input.flatten()).to(device=self.device).unsqueeze(0).float()

        with torch.no_grad():
            output = self.model(accel,
                                gyro,
                                odom_input,
                                patch)

            # print("desired : ", desired_odom)
            v, w = output.squeeze(0).cpu().numpy()

        # v, w = 1.0, 0.1

        print("Received Nav Command : ", msg.velocity, msg.velocity * msg.curvature)
        print("Output Nav Command : ", v, w)

        # populate with v and w
        self.nav_cmd.velocity = v
        self.nav_cmd.curvature = w / v
        self.nav_publisher.publish(self.nav_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ikd node')
    parser.add_argument('--model_path', default='models/visiontorchmodel.pt', type=str)
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


