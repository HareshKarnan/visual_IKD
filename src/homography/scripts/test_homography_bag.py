#!/usr/bin/env python3.6

import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
import rospy
import message_filters
import cv2
from scipy.spatial.transform import Rotation as R

def homography_camera_displacement(R1, R2, t1, t2, n1):
    R12 = R2 @ R1.T
    t12 = R2 @ (- R1.T @ t1) + t2
    # d is distance from plane to t1.
    d = np.linalg.norm(n1.dot(t1.T))

    H12 = R12 - ((t12 @ n1.T) / d)
    H12 /= H12[2, 2]
    return H12

def callback(odom, image):
    print(odom.pose.pose.position.z)
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
    # R_imu_world[0] = 0.5
    # R_imu_world[1] = 0.
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

    H12 = homography_camera_displacement(R1, R2, t1, t2, n1)
    homography_matrix = C_i @ H12 @ np.linalg.inv(C_i)
    homography_matrix /= homography_matrix[2, 2]

    img = np.fromstring(image.data, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    output = cv2.warpPerspective(img, homography_matrix, (1280, 720))

    # img = cv2.resize(img, (640, 360))
    # output = cv2.resize(output, (640, 360))
    cv2.imshow('disp', np.hstack((img, output)))
    cv2.waitKey(1)

def listener():
    rospy.init_node('test_homography_bag', anonymous=True)
    odom = message_filters.Subscriber('/camera/odom/sample', Odometry)
    image = message_filters.Subscriber('/webcam/image_raw/compressed', CompressedImage)
    # ts = message_filters.TimeSynchronizer([odom, image], 10)
    ts = message_filters.ApproximateTimeSynchronizer([odom, image], 10, 0.1, allow_headerless=False)
    ts.registerCallback(callback)

    rospy.spin()

if __name__ == '__main__':
    listener()
