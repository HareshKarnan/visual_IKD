#!/usr/bin/env python3.6
import os.path
from logging import root
import pickle
import numpy as np
import time
import rospy
import cv2
import matplotlib.pyplot as plt
from sensor_msgs.msg import CompressedImage, Joy, Imu
from nav_msgs.msg import Odometry
import message_filters
from termcolor import cprint
import yaml
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import signal
import subprocess

class ListenRecordData:
    def __init__(self, config_path, save_data_path, rosbag_play_process):
        self.data = []
        self.config_path = config_path
        self.save_data_path = save_data_path
        self.rosbag_play_process = rosbag_play_process

        with open(config_path, 'r') as f:
            cprint('Reading Config file.. ', 'yellow')
            self.config = yaml.safe_load(f)
            cprint('Parsed Config file successfully ', 'yellow', attrs=['blink'])
            print(self.config)

        # image = message_filters.Subscriber("/terrain_patch/compressed", CompressedImage)
        image = message_filters.Subscriber("/webcam/image_raw/compressed", CompressedImage)
        vectornavimu = message_filters.Subscriber("/vectornav/IMU", Imu)
        odom = message_filters.Subscriber('/camera/odom/sample', Odometry)
        accel = message_filters.Subscriber('/camera/accel/sample', Imu)
        gyro = message_filters.Subscriber('/camera/gyro/sample', Imu)
        joystick = message_filters.Subscriber('/joystick', Joy)
        ts = message_filters.ApproximateTimeSynchronizer([image, odom, joystick, accel, gyro, vectornavimu], 20, 0.05, allow_headerless=False)
        ts.registerCallback(self.callback)

        self.data = {'image': [], 'odom': [], 'joystick': [], 'accel': [], 'gyro': [], 'vectornav': []}

    def callback(self, image, odom, joystick, accel, gyro, vectornavimu):
        # print('Received messages :: ', image.header.seq)

        self.data['image'].append(image)
        self.data['odom'].append(odom)
        self.data['joystick'].append(joystick)
        self.data['accel'].append(accel)
        self.data['gyro'].append(gyro)
        self.data['vectornav'].append(vectornavimu)

    def save_data(self):
        # process joystick
        print('Processing joystick data')
        self.data = self.process_joystick_data(self.data, self.config)
        # prcess accel, gyro data
        print('Processing accel, gyro data')
        self.data = self.process_accel_gyro_data(self.data)
        # process bev image
        print('Processing bev image')
        self.data = self.process_bev_image(self.data)
        # process odom
        print('Processing odom data')
        self.data = self.process_odom_data(self.data)

        # save data
        cprint('Saving data.. ', 'yellow')
        pickle.dump(self.data, open(self.save_data_path, 'wb'))
        cprint('Saved data successfully ', 'yellow', attrs=['blink'])

        print('Number of data samples : ', len(self.data['image']))
        exit(0)

    def process_bev_image(self, data):
        for i in tqdm(range(len(data['image']))):
            bevimage = self.camera_imu_homography(data['vectornav'][i], data['image'][i])
            data['image'][i] = bevimage
        return data

    def process_odom_data(self, data):
        for i in tqdm(range(len(data['odom']))):
            odom = data['odom'][i]
            odom_np = np.array([odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])
            data['odom'][i] = odom_np
        return data

    @staticmethod
    def process_joystick_data(data, config):
        # process joystick
        last_speed = 0.0
        slipped_speeds = []
        for i in tqdm(range(len(data['joystick']))):
            data['joystick'][i] = data['joystick'][i].axes
            # print(data['joystick'][i])
            steer_joystick = -data['joystick'][i][0]
            drive_joystick = -data['joystick'][i][4]
            turbo_mode = data['joystick'][i][2] >= 0.9
            max_speed = turbo_mode * config['turbo_speed'] + (1 - turbo_mode) * config['normal_speed']
            speed = drive_joystick * max_speed
            steering_angle = steer_joystick * config['maxTurnRate']

            smooth_speed = max(speed, last_speed - config['commandInterval'] * config['accel_limit'])
            smooth_speed = min(smooth_speed, last_speed + config['commandInterval'] * config['accel_limit'])
            last_speed = smooth_speed
            erpm = config['speed_to_erpm_gain'] * smooth_speed + config['speed_to_erpm_offset']
            erpm_clipped = min(max(erpm, -config['erpm_speed_limit']), config['erpm_speed_limit'])
            clipped_speed = (erpm_clipped - config['speed_to_erpm_offset']) / config['speed_to_erpm_gain']

            servo = config['steering_to_servo_gain'] * steering_angle + config['steering_to_servo_offset']
            clipped_servo = min(max(servo, config['servo_min']), config['servo_max'])
            steering_angle = (clipped_servo - config['steering_to_servo_offset']) / config['steering_to_servo_gain']
            rot_vel = clipped_speed / config['wheelbase'] * np.tan(steering_angle)

            data['joystick'][i] = [clipped_speed, rot_vel]
        data['joystick'] = np.asarray(data['joystick'])

        return data

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
        # orientation_quat = [odom.pose.pose.orientation.x,
        #                     odom.pose.pose.orientation.y,
        #                     odom.pose.pose.orientation.z,
        #                     odom.pose.pose.orientation.w]

        orientation_quat = [imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w]

        # z_correction = odom.pose.pose.position.z

        C_i = np.array(
            [622.0649233612024, 0.0, 633.1717569157071, 0.0, 619.7990184421728, 368.0688607187958, 0.0, 0.0, 1.0]).reshape(
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
        output = output[420:520, 540:740]

        return output

    @staticmethod
    def process_accel_gyro_data(data):
        for i in tqdm(range(len(data['accel']))):
            accel = data['accel'][i].linear_acceleration
            gyro = data['gyro'][i].angular_velocity
            data['accel'][i] = [accel.x, accel.y, accel.z]
            data['gyro'][i] = [gyro.x, gyro.y, gyro.z]
        data['accel'] = np.asarray(data['accel'])
        data['gyro'] = np.asarray(data['gyro'])
        return data

if __name__ == '__main__':
    rospy.init_node('rosbag_data_recorder', anonymous=True)
    config_path = rospy.get_param('config_path')
    rosbag_path = rospy.get_param('rosbag_path')

    print('config_path: ', config_path)
    print('rosbag_path: ', rosbag_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError('Config file not found')
    if not os.path.exists(rosbag_path):
        raise FileNotFoundError('ROS bag file not found')

    # start a subprocess to run the rosbag
    rosbag_play_process = subprocess.Popen(['rosbag', 'play', rosbag_path, '-r', '1'])

    save_data_path = rosbag_path.replace('.bag', '_data.pkl')

    data_recorder = ListenRecordData(config_path=config_path,
                                     save_data_path=save_data_path,
                                     rosbag_play_process=rosbag_play_process)

    while not rospy.is_shutdown():
        # check if python subprocess is still running
        if rosbag_play_process.poll() is not None:
            print('rosbag_play process has stopped')
            print('Saving data..')
            data_recorder.save_data()
            print('Data saved successfully')

    rospy.spin()

