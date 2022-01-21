#!/usr/bin/python3
import argparse
import os

import rospy
import torch
torch.backends.cudnn.benchmark = True

import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage, Imu
from nav_msgs.msg import Odometry

import message_filters
from termcolor import cprint
import yaml
from scipy.spatial.transform import Rotation as R

from scripts.model import VisualIKDNet, SimpleIKDNet

import roslib
roslib.load_manifest('amrl_msgs')
from amrl_msgs.msg import AckermannCurvatureDriveMsg
import time
import torchgeometry as tgm
import kornia

PATCH_SIZE = 64
PATCH_EPSILON = 0.1 * PATCH_SIZE * PATCH_SIZE
ACTUATION_LATENCY = 0.1
C_i = np.array(
    [622.0649233612024, 0.0, 633.1717569157071, 0.0, 619.7990184421728, 368.0688607187958, 0.0, 0.0, 1.0]).reshape(
    (3, 3))
C_i_inv = np.linalg.inv(C_i)

C_i_torch = torch.from_numpy(C_i).float().cuda()
C_i_inv_torch = torch.from_numpy(C_i_inv).float().cuda()

R2 = R.from_euler("xyz", [0, 0, -90], degrees=True).as_matrix()
t2 = R2 @ np.array([-2.5, -0., 6.0]).reshape((3, 1))
R_cam_imu = R.from_euler("xyz", [90, -90, 0], degrees=True)

R2_torch = torch.from_numpy(R2).float().cuda()
t2_torch = torch.from_numpy(t2).float().cuda()

class LiveDataProcessor(object):
    def __init__(self, config_path, history_len, model_path):
        self.data = []
        self.config_path = config_path
        self.history_len = history_len

        with open(config_path, 'r') as f:
            cprint('Reading Config file.. ', 'yellow')
            self.config = yaml.safe_load(f)
            cprint('Parsed Config file successfully ', 'yellow', attrs=['blink'])
            print(self.config)


        self.accel_msgs = np.zeros((60, 3), dtype=np.float32)
        self.gyro_msgs = np.zeros((200, 3), dtype=np.float32)
        self.imu_msg = None
        self.data = {'accel': None, 'gyro': None, 'odom': None, 'patch': None}
        self.history_storage = {'bevimage': [], 'odom_msg': []}
        self.data_ready = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_counter = 0
        self.patch_history = 30

        print("Loading Model...")
        assert os.path.exists(model_path), "Model doesn't exist in the path"
        self.model = VisualIKDNet(input_size=3 * 60 + 3 * 200 + 3 + 2,
                                  output_size=2,
                                  hidden_size=32).to(device=self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("Loaded Model")

        rospy.Subscriber('/vectornav/IMU', Imu, self.imu_callback)
        rospy.Subscriber('/camera/accel/sample', Imu, self.accel_callback) #60 hz
        rospy.Subscriber('/camera/gyro/sample', Imu, self.gyro_callback) #200 hz
        rospy.Subscriber('/webcam/image_raw/compressed', CompressedImage, self.image_callback, queue_size=1)
        rospy.Subscriber('camera/odom/sample', Odometry, self.callback)

        self.nav_cmd = AckermannCurvatureDriveMsg()
        self.nav_cmd.velocity = 0.0
        self.nav_cmd.curvature = 0.0
        rospy.Subscriber('/ackermann_drive_init', AckermannCurvatureDriveMsg, self.navCallback)
        self.nav_publisher = rospy.Publisher('/ackermann_curvature_drive', AckermannCurvatureDriveMsg, queue_size=1)
        self.patch_observed = torch.tensor([True]).to(device=self.device).unsqueeze(0).float()
        self.callback_counter = 0

    def navCallback(self, msg):
        if not self.data_ready:
            print("Waiting for data processor initialization...Are all the necessary sensors running?")
            return
        odom_input = np.concatenate((self.data['odom'], np.array([msg.velocity, msg.velocity * msg.curvature])))
        odom_input = torch.tensor(odom_input.flatten()).to(device=self.device)

        accel = torch.tensor(self.accel_msgs.flatten()).to(device=self.device)
        gyro = torch.tensor(self.gyro_msgs.flatten()).to(device=self.device)

        with torch.no_grad():
            output = self.model(accel.unsqueeze(0).float(),
                                gyro.unsqueeze(0).float(),
                                odom_input.unsqueeze(0).float(),
                                self.data['patch'].float().cuda(),
                                self.patch_observed)
        v, w = output.squeeze(0).cpu().numpy()

        self.nav_cmd.velocity = v
        self.nav_cmd.curvature = w / v
        self.nav_publisher.publish(self.nav_cmd)

    def callback(self, odom):
        # populate the data dictionary

        self.data['odom'] = np.array([odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])
        self.latest_odom_msg = odom

        # check if 10 frames have been collected
        if len(self.history_storage['odom_msg']) < self.patch_history and len(self.history_storage['bevimage']) < self.patch_history:
            cprint('Not enough frames. Waiting for more frames to accumulate')
            return

        self.callback_counter += 1
        if self.callback_counter % 100 != 0: return

        # search for the patch in the past 3 frames
        found_patch, patch = False, None
        for j in range(self.patch_history):
            prev_image = self.history_storage['bevimage'][j]
            prev_odom = self.history_storage['odom_msg'][j]
            patch, patch_black_pct = self.get_patch_from_odom_delta(odom.pose.pose, prev_odom.pose.pose, odom.twist.twist, prev_image)
            if patch is not None:
                # patch has been found. Stop searching
                cprint('Found patch in the past 3 frames', 'green', attrs=['bold'])
                found_patch = True
                cv2.imshow('patch', patch)
                cv2.waitKey(1)
                patch = patch.astype(np.float32)/255.0
                patch = torch.from_numpy(patch).unsqueeze(0)
                self.patch_observed = torch.tensor([True]).to(device=self.device).unsqueeze(0).float()
                break

        if not found_patch:
            cprint('Could not find patch in the past 5 frames', 'red', attrs=['bold'])
            patch = self.bevimage_tensor[500:564, 613:677, :3].unsqueeze(0)
            self.patch_observed = torch.tensor([False]).to(device=self.device).unsqueeze(0).float()

        self.data['patch'] = patch.permute(0, 3, 1, 2)
        self.data_ready = True

    def image_callback(self, image):
        # TODO: instead of processing based on img_counter, use the odom distance
        self.img_counter += 1
        if self.img_counter % 10 == 0:
            self.history_storage['odom_msg'] = self.history_storage['odom_msg'][-self.patch_history-1:] + [self.latest_odom_msg]
            self.bevimage_tensor = self.camera_imu_homography(self.imu_msg, image)
            bevimage = ((self.bevimage_tensor/255.0).cpu().numpy()*255.0).astype(np.uint8)
            self.history_storage['bevimage'] = self.history_storage['bevimage'][-self.patch_history-1:] + [bevimage]
        return

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

    def camera_imu_homography(self, imu, image):
        orientation_quat = [imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w]

        R_imu_world = R.from_quat(orientation_quat)
        R_imu_world = R_imu_world.as_euler('xyz', degrees=True)
        R_imu_world[0], R_imu_world[1], R_imu_world[2] = -R_imu_world[0], R_imu_world[1], 0.

        R_imu_world = R.from_euler('xyz', R_imu_world, degrees=True)
        R1 = R_cam_imu * R_imu_world
        R1 = R1.as_matrix()

        # t1 = R1 @ np.array([0., 0., 0.5]).reshape((3, 1))
        # n1 = R1 @ np.array([0, 0, 1]).reshape((3, 1))
        # H12 = LiveDataProcessor.homography_camera_displacement(R1, t1, n1)
        # homography_matrix = C_i @ H12 @ np.linalg.inv(C_i)
        # homography_matrix /= homography_matrix[2, 2]

        H12 = LiveDataProcessor.homography_camera_displacement_torch(torch.from_numpy(R1).float().cuda())
        homography_matrix_torch = torch.matmul(C_i_torch, torch.matmul(H12, C_i_inv_torch))
        homography_matrix_torch = homography_matrix_torch / homography_matrix_torch[2, 2]
        homography_matrix_torch = homography_matrix_torch.reshape((1, 3, 3)).cuda()

        img = np.fromstring(image.data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # homoimg = cv2.warpPerspective(img.copy(), homography_matrix, (1280, 720))
        # homoimg = cv2.flip(homoimg, 1)
        # cv2.imshow('output', homoimg)
        # cv2.waitKey(1)

        img_torch = kornia.image_to_tensor(img.astype(np.uint8), keepdim=False).cuda()
        output_torch = kornia.geometry.transform.warp_perspective(img_torch.float(), homography_matrix_torch, (720, 1280)).squeeze(0)
        output_torch = torch.flip(output_torch, [2]).permute(1, 2, 0)

        # output_torch = output_torch/255.0
        # homoimg = output_torch.cpu().detach().numpy()
        # cv2.imshow('output', homoimg)
        # cv2.waitKey(1)

        return output_torch

    def imu_callback(self, msg):
        self.imu_msg = msg

    @staticmethod
    def homography_camera_displacement(R1, t1, n1):
        R12 = R2 @ R1.T
        t12 = R2 @ (- R1.T @ t1) + t2
        # d is distance from plane to t1.
        d = np.linalg.norm(n1.dot(t1.T))

        H12 = R12 - ((t12 @ n1.T) / d)
        H12 /= H12[2, 2]
        return H12

    @staticmethod
    def homography_camera_displacement_torch(R1):
        t1 = torch.matmul(R1, torch.tensor([0., 0., 0.5]).float().cuda()).reshape((3, 1)).cuda()
        n1 = torch.matmul(R1, torch.tensor([0, 0, 1]).float().cuda()).reshape((3, 1)).cuda()

        R12 = torch.matmul(R2_torch, R1.T)
        t12 = torch.matmul(R2_torch, torch.matmul(- R1.T, t1)) + t2_torch
        d = torch.norm(torch.mm(n1, t1.T))

        H12 = R12 - (torch.matmul(t12, n1.T) / d)
        H12 = H12 / H12[2, 2]
        return H12

    @staticmethod
    def get_patch_from_odom_delta(curr_pos, prev_pos, curr_vel, prev_image):
        curr_pos_np = np.array([curr_pos.position.x, curr_pos.position.y, 1])
        prev_pos_transform = np.zeros((3, 3))
        z_angle = R.from_quat(
            [prev_pos.orientation.x, prev_pos.orientation.y, prev_pos.orientation.z, prev_pos.orientation.w]).as_euler(
            'xyz', degrees=False)[2]
        prev_pos_transform[:2, :2] = R.from_euler('xyz', [0, 0, z_angle]).as_matrix()[:2, :2]  # figure this out
        prev_pos_transform[:, 2] = np.array([prev_pos.position.x, prev_pos.position.y, 1]).reshape((3))
        inv_pos_transform = LiveDataProcessor.affineinverse(prev_pos_transform)

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

        persp = cv2.getPerspectiveTransform(np.float32(patch_corners_image_frame),
                                            np.float32([[0, 0], [63, 0], [63, 63], [0, 63]]))

        patch = cv2.warpPerspective(
            prev_image,
            persp,
            (64, 64)
        )

        zero_count = np.logical_and(np.logical_and(patch[:, :, 0] == 0, patch[:, :, 1] == 0), patch[:, :, 2] == 0)

        if np.sum(zero_count) > PATCH_EPSILON:
            return None, 1.0

        return patch, (np.sum(zero_count) / (64. * 64.))

    @staticmethod
    def affineinverse(M):
        tmp = np.hstack((-M[:2, :2], -M[:2, :2] @ M[:2, 2].reshape((2, 1))))
        return np.vstack((tmp, np.array([0, 0, 1])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ikd node')
    parser.add_argument('--model_path', default='models/visiontorchmodel.pt', type=str)
    parser.add_argument('--input_topic', default='/ackermann_drive_init', type=str)
    parser.add_argument('--output_topic', default='/ackermann_curvature_drive',  type=str)
    parser.add_argument('--config_path', type=str, default="config/alphatruck.yaml")
    parser.add_argument('--history_len', type=int, default=1)
    args = parser.parse_args()

    rospy.init_node('ikd_node', anonymous=True)

    data_processor = LiveDataProcessor(args.config_path, args.history_len, args.model_path)
    # node = IKDNode(data_processor, args.model_path, args.history_len, args.input_topic, args.output_topic)

    while not rospy.is_shutdown():
        rospy.spin()


