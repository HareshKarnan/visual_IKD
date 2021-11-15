import rosbag
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Imu, Joy
import argparse
from ros_numpy import numpify
import numpy as np
import cv2
from termcolor import cprint
from tqdm import tqdm
import time
import pickle
import yaml
from scripts.utils import \
    parse_bag_with_img, \
    filter_data, \
    process_joystick_data, \
    process_trackingcam_data, \
    process_accel_gyro_data

from scipy.spatial.transform import Rotation as R
from tmp.test_homography import create_transformation_matrix

keys = ['rgb', 'odom', 'accel', 'gyro', 'joystick']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rosbag parser')
    parser.add_argument('--rosbag_path', type=str, default='data/ahgroad_new.bag')
    parser.add_argument('--frequency', type=int, default=20)
    parser.add_argument('--max_time', type=int, default=20)
    parser.add_argument('--config_path', type=str, default="config/alphatruck.yaml")
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        cprint('Reading Config file.. ', 'yellow')
        config = yaml.safe_load(f)
        cprint('Parsed Config file successfully ', 'yellow', attrs=['blink'])
        print(config)

    topics_to_read = [
        '/camera/odom/sample',
        '/joystick',
        '/camera/accel/sample',
        '/camera/gyro/sample',
        '/webcam/image_raw/compressed'
    ]

    # parse all data from the rosbags
    data, total_time = parse_bag_with_img(args.rosbag_path, topics_to_read, max_time=args.max_time)
    # set the time intervals
    times = np.linspace(0, int(total_time), args.frequency * int(total_time) + 1)
    # filter data based on time intervals
    filtered_data = filter_data(data, times, keys, viz_images=False)
    print('# filtered data points : ', len(filtered_data['rgb']))
    # process joystick data
    filtered_data = process_joystick_data(filtered_data, config=config)
    # process tracking cam data
    filtered_data = process_trackingcam_data(filtered_data)
    # process accel and gyro data
    filtered_data = process_accel_gyro_data(filtered_data)

    for i in range(len(filtered_data['rgb'])):
        image = filtered_data['rgb'][i]
        v, w = filtered_data['joystick'][i]
        x_vel, y_vel, w_vel, x_pos, y_pos, z_pos, w_pos = filtered_data['odom'][i]

        print(x_pos, y_pos, z_pos, w_pos)
        # cv2.imwrite('tmp/img0.png', image)
        # np.save('tmp/img0orientation.npy', np.asarray([x_pos, y_pos, z_pos, w_pos]))
        # break

        # image = cv2.putText(image, 'v : ' + str(v), (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # image = cv2.putText(image, 'w : ' + str(w), (500, 530), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #
        # image = cv2.putText(image, 'x_vel : ' + str(x_vel), (500, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # image = cv2.putText(image, 'y_vel : ' + str(y_vel), (500, 630), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # image = cv2.putText(image, 'w_vel : ' + str(w_vel), (500, 660), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.imshow('disp', image)
        P_odom = create_transformation_matrix([x_pos, y_pos, z_pos, w_pos])
        P_odom[2, 3] = 0.2


        time.sleep(0.05)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    import matplotlib.pyplot as plt
    time_axes = np.arange(0, len(filtered_data['rgb']))
    fig, axs = plt.subplots(2)
    axs[0].plot(times, filtered_data['odom'][:, 0],'-b' ,label='odom_x')
    axs[0].plot(times, filtered_data['joystick'][:, 0], '--b',label='joy_x')
    axs[0].legend()
    # plt.plot(time_axes, filtered_data['odom'][:, 1], label='odom_y')
    axs[1].plot(times, filtered_data['odom'][:, 2], '-r', label='odom_w')
    axs[1].plot(times, filtered_data['joystick'][:, 1], '--r',label='joy_w')
    axs[1].legend()

    plt.xlabel('Time')
    plt.show()






















