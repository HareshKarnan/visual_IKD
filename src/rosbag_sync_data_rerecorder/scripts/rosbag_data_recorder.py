#!/usr/bin/env python3
import os.path
import copy
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
import subprocess

PATCH_SIZE = 64
PATCH_EPSILON = 0.2 * PATCH_SIZE * PATCH_SIZE
ACTUATION_LATENCY = 0.1
BATCH_SIZE = 160000

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

        image = message_filters.Subscriber("/webcam/image_raw/compressed", CompressedImage)
        vectornavimu = message_filters.Subscriber("/vectornav/IMU", Imu)
        odom = message_filters.Subscriber('/camera/odom/sample', Odometry)
        joystick = message_filters.Subscriber('/joystick', Joy)
        ts = message_filters.ApproximateTimeSynchronizer([image, odom, joystick, vectornavimu], 10, 0.05, allow_headerless=True)
        ts.registerCallback(self.callback)
        self.batch_idx = 0
        self.counter = 0

        # subscribe to accel and gyro topics
        rospy.Subscriber('/camera/accel/sample', Imu, self.accel_callback) # 60hz
        rospy.Subscriber('/camera/gyro/sample', Imu, self.gyro_callback) # 200hz

        self.msg_data = {
            'image_msg': [],
            'src_image': [],
            'odom_msg': [],
            'joystick_msg': [],
            'accel_msg': [],
            'gyro_msg': [],
            'vectornav': [],
        }

        self.open_thread_lists = []

        self.accel_msgs = np.zeros((60, 3), dtype=np.float32)
        self.gyro_msgs = np.zeros((200, 3), dtype=np.float32)

    def callback(self, image, odom, joystick, vectornavimu):
        print('Received messages :: ', self.counter, ' ___')
    
        self.msg_data['image_msg'].append(image)
        self.msg_data['odom_msg'].append(odom)
        self.msg_data['joystick_msg'].append(joystick)
        self.msg_data['vectornav'].append(vectornavimu)
        self.msg_data['accel_msg'].append(self.accel_msgs.flatten())
        self.msg_data['gyro_msg'].append(self.gyro_msgs.flatten())
        self.counter += 1

        # if (len(self.msg_data['image_msg']) > BATCH_SIZE):
        #     for key in self.msg_data.keys():
        #         self.msg_data[key] = self.msg_data[key][-BATCH_SIZE:]

    def accel_callback(self, msg):
        # add to queue self.accel_msgs
        self.accel_msgs = np.roll(self.accel_msgs, -1, axis=0)
        self.accel_msgs[-1] = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

    def gyro_callback(self, msg):
        # add to queue self.gyro_msgs
        self.gyro_msgs = np.roll(self.gyro_msgs, -1, axis=0)
        self.gyro_msgs[-1] = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])


    def save_data(self, msg_data, batch_idx):
        data = {}
        # process joystick
        print('Processing joystick data')
        data['joystick'] = self.process_joystick_data(msg_data, self.config)
        del msg_data['joystick_msg']

        # process accel, gyro data
        print('Processing accel, gyro data')
        # data['accel'], data['gyro'] = self.process_accel_gyro_data(msg_data)
        data['accel'], data['gyro'] = msg_data['accel_msg'], msg_data['gyro_msg']
        del msg_data['accel_msg']
        del msg_data['gyro_msg']

        # process odom
        print('Processing odom data')
        data['odom'] = self.process_odom_vel_data(msg_data)
        data['vectornav'] = msg_data['vectornav']

        # # convert egocentric image view into a bird's eye view based on the IMU data
        # print('Processing bev image')
        # data['image'] = self.process_bev_image(msg_data)
        #
        # print('Processing patches')
        # data['patches'] = self.process_patches(msg_data, data)

        data['patches'] = self.process_bev_image_and_patches(msg_data)

        del msg_data['image_msg']
        del msg_data['odom_msg']
        del msg_data['vectornav']

        print("truncating data")
        for i in range(len(data['odom'])-1, -1, -1):
            if (i not in data['patches']):
                data['joystick'].pop(i)
                data['accel'].pop(i)
                data['gyro'].pop(i)
                data['odom'].pop(i)
                data['vectornav'].pop(i)
        assert(len(data['odom']) == len(data['patches'].keys()))
        patches = []
        sorted_keys = sorted(data['patches'].keys())
        for i in range(len(sorted_keys)):
            patches.append(data['patches'][sorted_keys[i]])
        data['patches'] = patches

        # dont save vectornav
        del data['vectornav']

        if len(data['odom']) > 0:
            # save data
            cprint('Saving data...{}'.format(len(data['odom'])), 'yellow')
            path = os.path.join(self.save_data_path, 'data_{}.pkl'.format(batch_idx))
            pickle.dump(data, open(path, 'wb'))
            cprint('Saved data successfully ', 'yellow', attrs=['blink'])

    def process_bev_image_and_patches(self, msg_data):
        processed_data = {'image':[]}
        msg_data['patches'] = {}

        for i in tqdm(range(len(msg_data['image_msg']))):
            bevimage, _ = ListenRecordData.camera_imu_homography(msg_data['vectornav'][i], msg_data['image_msg'][i])
            processed_data['image'].append(bevimage)

            # now find the patches for this image
            curr_odom = msg_data['odom_msg'][i]

            found_patch = False
            for j in range(i, max(i-30, 0), -1):
                prev_image = processed_data['image'][j]
                prev_odom = msg_data['odom_msg'][j]
                # cv2.imshow('src_image', processed_data['src_image'][i])
                patch, patch_black_pct, curr_img, vis_img = ListenRecordData.get_patch_from_odom_delta(
                    curr_odom.pose.pose, prev_odom.pose.pose, curr_odom.twist.twist,
                    prev_odom.twist.twist, prev_image, processed_data['image'][i])
                if patch is not None:
                    found_patch = True
                    if i not in msg_data['patches']:
                        msg_data['patches'][i] = []
                    msg_data['patches'][i].append(patch)
                if len(msg_data['patches'][i]) > 10: break
            if not found_patch:
                print("Unable to find patch for idx: ", i)
            # else:
            #     # show the image and the patches
            #     cv2.imshow('image', processed_data['image'][i])
            #     for patch in msg_data['patches'][i]:
            #         cv2.imshow('patch', patch)
            #         cv2.waitKey(0)

            # remove the i-30th image from RAM
            if i > 30:
                processed_data['image'][i-29] = None

        return msg_data['patches']


    def process_bev_image(self, data):
        bevimages = []
        for i in tqdm(range(len(data['image_msg']))):
            bevimage, _ = ListenRecordData.camera_imu_homography(data['vectornav'][i], data['image_msg'][i])
            bevimages.append(bevimage)
        return bevimages

    @staticmethod
    def process_patches(data, processed_data):
        patches = {}
        for i in range(len(processed_data['image'])):
            curr_odom = data['odom_msg'][i]
            found_patch = False
            for j in range(i, max(i - 30, 0), -2):
                prev_image = processed_data['image'][j]
                prev_odom = data['odom_msg'][j]
                # cv2.imshow('src_image', processed_data['src_image'][i])
                patch, patch_black_pct, curr_img, vis_img = ListenRecordData.get_patch_from_odom_delta(curr_odom.pose.pose, prev_odom.pose.pose, curr_odom.twist.twist,
                                                                                                       prev_odom.twist.twist, prev_image, processed_data['image'][i])
                if patch is not None:
                    found_patch = True
                    if i not in patches:
                        patches[i] = []
                    patches[i].append(patch)
            if not found_patch:
                print("Unable to find patch for idx: ", i)
            else:
                # show the image and the patches
                cv2.imshow('image', processed_data['image'][i])
                for patch in patches[i]:
                    cv2.imshow('patch', patch)
                    cv2.waitKey(0)

        return patches

    @staticmethod
    def get_patch_from_odom_delta(curr_pos, prev_pos, curr_vel, prev_vel, prev_image, curr_image):
        curr_pos_np = np.array([curr_pos.position.x, curr_pos.position.y, 1])
        prev_pos_transform = np.zeros((3, 3))
        z_angle = R.from_quat([prev_pos.orientation.x, prev_pos.orientation.y, prev_pos.orientation.z, prev_pos.orientation.w]).as_euler('xyz', degrees=False)[2]
        prev_pos_transform[:2, :2] = R.from_euler('xyz', [0, 0, z_angle]).as_matrix()[:2,:2] # figure this out
        prev_pos_transform[:, 2] = np.array([prev_pos.position.x, prev_pos.position.y, 1]).reshape((3))

        inv_pos_transform = np.linalg.inv(prev_pos_transform)
        curr_z_angle = R.from_quat([curr_pos.orientation.x, curr_pos.orientation.y, curr_pos.orientation.z, curr_pos.orientation.w]).as_euler('xyz', degrees=False)[2]
        curr_z_rotation = R.from_euler('xyz', [0, 0, curr_z_angle]).as_matrix()
        projected_loc_np  = curr_pos_np + ACTUATION_LATENCY * (curr_z_rotation @ np.array([curr_vel.linear.x, curr_vel.linear.y, 0]))

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

        # draw the patch rectangle
        cv2.line(
            vis_img,
            (patch_corners_image_frame[0][0], patch_corners_image_frame[0][1]),
            (patch_corners_image_frame[1][0], patch_corners_image_frame[1][1]),
            (0, 255, 0),
            2
        )
        cv2.line(
            vis_img,
            (patch_corners_image_frame[1][0], patch_corners_image_frame[1][1]),
            (patch_corners_image_frame[2][0], patch_corners_image_frame[2][1]),
            (0, 255, 0),
            2
        )
        cv2.line(
            vis_img,
            (patch_corners_image_frame[2][0], patch_corners_image_frame[2][1]),
            (patch_corners_image_frame[3][0], patch_corners_image_frame[3][1]),
            (0, 255, 0),
            2
        )
        cv2.line(
            vis_img,
            (patch_corners_image_frame[3][0], patch_corners_image_frame[3][1]),
            (patch_corners_image_frame[0][0], patch_corners_image_frame[0][1]),
            (0, 255, 0),
            2
        )
        # draw movement vector
        mov_start = curr_pos_np
        mov_end = curr_pos_np + (projected_loc_np - curr_pos_np) * 5

        head_start = curr_pos_np
        head_end = curr_pos_np + curr_z_rotation @ np.array([0.25, 0, 0])

        ListenRecordData.draw_arrow(mov_start, mov_end, (255, 0, 0), inv_pos_transform, CENTER, vis_img)
        ListenRecordData.draw_arrow(head_start, head_end, (0, 0, 255), inv_pos_transform, CENTER, vis_img)

        projected_loc_prev_frame = inv_pos_transform @ projected_loc_np
        scaled_projected_loc = (projected_loc_prev_frame * 200).astype(np.int)
        projected_loc_image_frame = CENTER + np.array((-scaled_projected_loc[1], -scaled_projected_loc[0]))
        cv2.circle(vis_img, (projected_loc_image_frame[0], projected_loc_image_frame[1]), 3, (0, 255, 255))

        persp = cv2.getPerspectiveTransform(np.float32(patch_corners_image_frame), np.float32([[0, 0], [63, 0], [63, 63], [0, 63]]))

        patch = cv2.warpPerspective(
            prev_image,
            persp,
            (64, 64)
        )

        zero_count = np.logical_and(np.logical_and(patch[:, :, 0] == 0, patch[:, :, 1] == 0), patch[:, :, 2] == 0)

        if np.sum(zero_count) > PATCH_EPSILON:
            return None, 1.0, None, None

        return patch, (np.sum(zero_count) / (64. * 64.)), curr_image, vis_img

    @staticmethod
    def draw_arrow(arrow_start, arrow_end, color, inv_pos_transform, CENTER, vis_img):
        arrow_start_prev_frame = inv_pos_transform @ arrow_start
        arrow_end_prev_frame = inv_pos_transform @ arrow_end
        scaled_arrow_start = (arrow_start_prev_frame * 200).astype(np.int)
        scaled_arrow_end = (arrow_end_prev_frame * 200).astype(np.int)
        arrow_start_image_frame = CENTER + np.array((-scaled_arrow_start[1], -scaled_arrow_start[0]))
        arrow_end_image_frame = CENTER + np.array((-scaled_arrow_end[1], -scaled_arrow_end[0]))

        cv2.arrowedLine(
            vis_img,
            (arrow_start_image_frame[0], arrow_start_image_frame[1]),
            (arrow_end_image_frame[0], arrow_end_image_frame[1]),
            color,
            3
        )

    @staticmethod
    def process_odom_vel_data(data):
        odoms = []
        for i in range(len(data['odom_msg'])):
            odom = data['odom_msg'][i]
            odom_np = np.array([odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])
            odoms.append(odom_np)
        return odoms

    @staticmethod
    def process_joystick_data(data, config):
        # process joystick
        last_speed = 0.0
        joystick_data = []
        for i in range(len(data['joystick_msg'])):
            datum = data['joystick_msg'][i].axes # TODO use a previous joystick command based on actuation LATENCY
            # print(data['joystick'][i])
            steer_joystick = -datum[0]
            drive_joystick = -datum[4]
            turbo_mode = datum[2] >= 0.9
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

            datum = [clipped_speed, rot_vel]
            joystick_data.append(datum)

        return joystick_data

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
    def camera_imu_homography(imu, image):
        orientation_quat = [imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w]
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

        H12 = ListenRecordData.homography_camera_displacement(R1, R2, t1, t2, n1)
        homography_matrix = C_i @ H12 @ np.linalg.inv(C_i)
        homography_matrix /= homography_matrix[2, 2]

        img = np.fromstring(image.data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        output = cv2.warpPerspective(img, homography_matrix, (1280, 720))
        # flip output horizontally
        output = cv2.flip(output, 1)

        return output, img

    @staticmethod
    def process_accel_gyro_data(data):
        accel_data = []
        gyro_data = []
        for i in range(len(data['accel_msg'])):
            accel = data['accel_msg'][i].linear_acceleration
            gyro = data['gyro_msg'][i].angular_velocity
            accel_data.append(np.asarray([accel.x, accel.y, accel.z]))
            gyro_data.append(np.asarray([gyro.x, gyro.y, gyro.z]))
        return accel_data, gyro_data

if __name__ == '__main__':
    rospy.init_node('rosbag_data_recorder', anonymous=True)
    config_path = rospy.get_param('config_path')
    rosbag_path = rospy.get_param('rosbag_path')
    save_data_path = rospy.get_param('out_path')

    print('config_path: ', config_path)
    print('rosbag_path: ', rosbag_path)
    if not save_data_path:
        save_data_path = rosbag_path.replace('.bag', '_data.pkl')
    print('save_data_path: ', save_data_path)
    os.makedirs(save_data_path, exist_ok=True)

    if not os.path.exists(config_path):
        raise FileNotFoundError('Config file not found')
    if not os.path.exists(rosbag_path):
        raise FileNotFoundError('ROS bag file not found')

    # start a subprocess to run the rosbag
    rosbag_play_process = subprocess.Popen(['rosbag', 'play', rosbag_path, '-r', '1'])

    data_recorder = ListenRecordData(config_path=config_path,
                                     save_data_path=save_data_path,
                                     rosbag_play_process=rosbag_play_process)

    while not rospy.is_shutdown():
        # check if python subprocess is still running
        if rosbag_play_process.poll() is not None:
            print('rosbag_play process has stopped')

            # # check if there is some data left to be stored in the buffer
            # if data_recorder.counter % BATCH_SIZE > 0:
            #     print('There is some data left in the buffer with length :', data_recorder.counter % BATCH_SIZE)
            #     for key in data_recorder.msg_data.keys():
            #         data_recorder.msg_data[key] = data_recorder.msg_data[key][-data_recorder.counter % BATCH_SIZE:]
            #     data_recorder.save_data(copy.deepcopy(data_recorder.msg_data), data_recorder.batch_idx + 1)
            #     print('Buffer data has been stored')
            # else:
            #     print('No data left in the buffer')
            #
            # print('waiting for ', len(data_recorder.open_thread_lists), ' threads to finish ... ')
            # for threads in tqdm(data_recorder.open_thread_lists):
            #     threads.join()

            data_recorder.save_data(copy.deepcopy(data_recorder.msg_data), data_recorder.batch_idx + 1)
            exit(0)

    rospy.spin()


