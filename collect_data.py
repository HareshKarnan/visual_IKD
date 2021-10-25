"""
Script that records sensor information using rosbag record
"""

import os
import rospy
from sensor_msgs.msg import Joy
import signal
import subprocess

# rgb_topic = '/camera/rgb/image_raw/compressed'
rgb_topic = '/webcam/image_raw/compressed'
depth_topic = '/camera/depth/image_raw/compressed'
odom_topic = '/camera/odom/sample'
gyro_topic = '/camera/gyro/sample'
accel_topic = '/camera/accel/sample'
joystick_topic = '/joystick'
tf_topic = '/tf'
tfstatic_topic = '/tf_static'
vescdrive_topic = '/vesc_drive'


class DataCollector:
    def __init__(self):
        self.setup_subscribers()
        self.button_x, self.button_o = 0, 0
        self.pro = None

    def setup_subscribers(self):
        self.joy_sub = rospy.Subscriber('/joystick', Joy, self.joy_cb)
        print('Subscribers setup successfully')

    def joy_cb(self, data):
        self.button_x = data.buttons[0]
        self.button_o = data.buttons[1]

        if self.button_o == 1 and self.pro == None:
            print('Starting data collection')
            # start data collection
            cmd = 'exec rosbag record ' + \
                rgb_topic + ' '+\
                odom_topic + ' '+\
                gyro_topic + ' '+\
                accel_topic + ' '+\
                joystick_topic + ' '+\
                tf_topic + ' '+\
                tfstatic_topic +' '+\
                vescdrive_topic

            self.pro = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            print('Data collection started')


        if self.button_x == 1:

            if self.pro is None:
                print('Press button O before killing the process with X')

            else:
                # kill the process
                print('Ending data collection')
                # os.kill(os.getpid(self.pro.pid), signal.SIGINT)
                self.pro.kill()
                print('Ended data collection')
                self.pro = None


if __name__ == "__main__":
    rospy.init_node('rosbag_recorder_trigger', anonymous=True)
    recorder = DataCollector()
    rospy.spin()


