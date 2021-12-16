import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

normal_speed = 1.0
turbo_speed = 3.0
accel_limit = 6.0
maxTurnRate = 0.25
commandInterval = 1.0 / 20
speed_to_erpm_gain = 5171
speed_to_erpm_offset = 180.0
erpm_speed_limit = 22000
steering_to_servo_gain = -0.9015
steering_to_servo_offset = 0.553
servo_min = 0.05
servo_max = 0.95
wheelbase = 0.324


def get_value_at_time(time, times, values):
    if time < times[0] or time > times[-1]:
        return 0.0
    left, right = 0, len(times) - 1
    while right - left > 1:
        mid = (left + right) // 2
        if time > times[mid]:
            left = mid
        elif time < times[mid]:
            right = mid
        else:
            return values[mid]
    return values[left] + (values[right] - values[left]) * (time - times[left]) / (times[right] - times[left])


subfolders = ['data/ahg_road']
for subfolder in subfolders:
    # Extract commanded velocity & angular velocity
    data_frame = pd.read_csv(subfolder + "/_slash_joystick.csv")

    secs = data_frame["secs"].to_numpy()
    nsecs = data_frame["nsecs"].to_numpy()
    joystick_times = secs + nsecs / 1e9 - secs[0]

    axes_strings = data_frame["axes"].to_numpy()
    axes = []
    for ax in axes_strings:
        ax = ax[1:-1]
        ax = ax.split(", ")
        ax = [float(a) for a in ax]
        axes.append(ax)
    axes = np.array(axes)

    steer_joystick = -axes[:, 0]
    drive_joystick = -axes[:, 4]
    turbo_mode = axes[:, 2] >= 0.9
    max_speed = turbo_mode * turbo_speed + (1 - turbo_mode) * normal_speed
    speed = drive_joystick * max_speed
    steering_angle = steer_joystick * maxTurnRate

    last_speed = 0.0
    clipped_speeds = []
    for s in speed:
        smooth_speed = max(s, last_speed - commandInterval * accel_limit)
        smooth_speed = min(smooth_speed, last_speed + commandInterval * accel_limit)
        last_speed = smooth_speed
        erpm = speed_to_erpm_gain * smooth_speed + speed_to_erpm_offset
        erpm_clipped = min(max(erpm, -erpm_speed_limit), erpm_speed_limit)
        clipped_speed = (erpm_clipped - speed_to_erpm_offset) / speed_to_erpm_gain
        clipped_speeds.append(clipped_speed)
    clipped_speeds = np.array(clipped_speeds)

    servo = steering_to_servo_gain * steering_angle + steering_to_servo_offset
    clipped_servo = np.fmin(np.fmax(servo, servo_min), servo_max)
    steering_angle = (clipped_servo - steering_to_servo_offset) / steering_to_servo_gain

    rot_vel = clipped_speeds / wheelbase * np.tan(steering_angle)

    joystick_data = (joystick_times, clipped_speeds, rot_vel)

    # Extract tracking cam linear and angular velocity data
    data_frame = pd.read_csv(subfolder + "/_slash_camera_slash_odom_slash_sample.csv")

    secs = data_frame["secs"].to_numpy()
    nsecs = data_frame["nsecs"].to_numpy()
    times = secs + nsecs / 1e9 - secs[0]

    x_vel = data_frame["x.2"].to_numpy()
    y_vel = data_frame["y.2"].to_numpy()
    velocities = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angular_vels = data_frame["z.3"].to_numpy()

    tracking_cam_data = (times, velocities, angular_vels, x_vel, y_vel)

    # Generate dataset
    end_time = int(min(joystick_times[-1], times[-1]))
    time_points = np.linspace(0, end_time, end_time * 20 + 1)
    cmd_v, cmd_w, vx, vy, w = [], [], [], [], []
    f = open(subfolder + "/dataset.txt", "w")
    f.write("t\tcmd_v\tcmd_w\tvx\tvy\tw\n")
    for t in time_points:
        cmd_v.append(get_value_at_time(t, joystick_data[0], joystick_data[1]))
        cmd_w.append(get_value_at_time(t, joystick_data[0], joystick_data[2]))
        vx.append(get_value_at_time(t, tracking_cam_data[0], tracking_cam_data[3]))
        vy.append(get_value_at_time(t, tracking_cam_data[0], tracking_cam_data[4]))
        w.append(get_value_at_time(t, tracking_cam_data[0], tracking_cam_data[2]))

        f.write(str(t) + "\t")
        f.write(str(cmd_v[-1]) + "\t")
        f.write(str(cmd_w[-1]) + "\t")
        f.write(str(vx[-1]) + "\t")
        f.write(str(vy[-1]) + "\t")
        f.write(str(w[-1]) + "\n")
    f.close()
    cmd_v, cmd_w, vx, vy, w = np.array(cmd_v), np.array(cmd_w), np.array(vx), np.array(vy), np.array(w)

    # Plot data
    fig, ax = plt.subplots()
    ax.plot(time_points, vx, label="vx")
    ax.plot(time_points, vy, label="vy")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("vel (m/s)")
    ax.set_title("VX and VY")
    ax.legend()
    plt.savefig(subfolder + "/vx_and_vy.png")

    fig, ax = plt.subplots()
    ax.plot(time_points, (vx ** 2 + vy ** 2) ** 0.5, label="Actual Abs Vel")
    ax.plot(time_points, cmd_v, label="Commanded Abs Vel")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("vel (m/s)")
    ax.set_title("Commanded vs Actual Abs Vel")
    ax.legend()
    plt.savefig(subfolder + "/commanded_vs_actual_abs_vel.png")

    fig, ax = plt.subplots()
    ax.plot(time_points, w, label="Actual Angular Vel")
    ax.plot(time_points, cmd_w, label="Commanded Angular Vel")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angular vel (rad/s)")
    ax.set_title("Commanded vs Actual Angular Vel")
    ax.legend()
    plt.savefig(subfolder + "/commanded_vs_actual_angular_vel.png")