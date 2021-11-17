import numpy as np
import cv2
from tqdm import tqdm
from termcolor import cprint
import rosbag
import random
from PIL import Image, ImageOps, ImageFilter

def parse_bag_with_img(bagfile_path, topics_to_read=None, max_time=None):
    bag = rosbag.Bag(bagfile_path, 'r')

    assert topics_to_read is not None, "Pass the list of topics to read"
    n = 0

    odom_list, joystick_list = [], []
    accel_list, gyro_list = [], []
    image_list = []
    cprint('Parsing the rosbag...', 'red')

    for topic, msg, t in tqdm(bag.read_messages(topics=topics_to_read)):
        if n==0:
            start_time = t.to_sec()
            n+=1
        time = t.to_sec() - start_time

        if topic == '/camera/odom/sample':
            odom_list.append((time, msg))
        elif topic == '/joystick':
            joystick_list.append((time, msg))
        elif topic == '/camera/accel/sample':
            accel_list.append((time, msg))
        elif topic == '/camera/gyro/sample':
            gyro_list.append((time, msg))
        elif topic == '/webcam/image_raw/compressed':
            # convert compressed image to opencv-numpy image
            img = np.fromstring(msg.data, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            image_list.append((time, img))

        if (max_time is not None) and (max_time != -1) and max_time<time: break

    data = {
        'odom': odom_list,
        'accel': accel_list,
        'gyro': gyro_list,
        'rgb': image_list,
        'joystick': joystick_list
    }

    bag.close()
    print('total time : ', time)
    return data, time

def fetch_nearest_time(time, data_list, key):
    min, max = 0, len(data_list[key])-1
    while min<max:
        curr = int((min + max)/2)
        if data_list[key][curr][0] < time:
            min = curr+1
        elif data_list[key][curr][0] > time:
            max = curr-1
        else:
            return data_list[key][curr][1], curr
    return data_list[key][curr][1], curr

def filter_data(data, times, keys, viz_images=True):
    # filter the data based on time frequency
    cprint('Filtering the data based on time frequency..', 'red')

    data_filtered = {}
    for key in keys:
        data_filtered[key] = []

    for time in tqdm(times):
        for key in keys:
            data_at_time, time_t = fetch_nearest_time(time, data, key)
            data_filtered[key].append(data_at_time)

    if viz_images:
        for i in range(len(data_filtered['rgb'])):
            cv2.imshow('disp', data_filtered['rgb'][i])
            cv2.waitKey(100)

    return data_filtered

def process_joystick_data(data, config):
    # process joystick
    last_speed = 0.0
    slipped_speeds = []
    for i in range(len(data['joystick'])):
        data['joystick'][i] = data['joystick'][i].axes
        # print(data['joystick'][i])
        steer_joystick = -data['joystick'][i][0]
        drive_joystick = -data['joystick'][i][4]
        turbo_mode = data['joystick'][i][2] >= 0.9
        max_speed = turbo_mode * config['turbo_speed'] + (1-turbo_mode) * config['normal_speed']
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

def process_trackingcam_data(data):
    for i in range(len(data['odom'])):
        twist = data['odom'][i].twist.twist
        pose = data['odom'][i].pose.pose
        x_vel = twist.linear.x
        y_vel = twist.linear.y
        angular_vel = twist.angular.z
        x_pos = pose.orientation.x
        y_pos = pose.orientation.y
        z_pos = pose.orientation.z
        w_pos = pose.orientation.w
        data['odom'][i] = [x_vel, y_vel, angular_vel, x_pos, y_pos, z_pos, w_pos]
    data['odom'] = np.asarray(data['odom'])
    return data

def process_accel_gyro_data(data):
    for i in range(len(data['rgb'])):
        accel = data['accel'][i].linear_acceleration
        gyro = data['gyro'][i].angular_velocity
        data['accel'][i] = [accel.x, accel.y, accel.z]
        data['gyro'][i] = [gyro.x, gyro.y, gyro.z]
    data['accel'] = np.asarray(data['accel'])
    data['gyro'] = np.asarray(data['gyro'])
    return data

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img