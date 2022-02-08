import numpy as np
import matplotlib.pyplot as plt
import rosbag
from geometry_msgs.msg import TwistStamped

# Load data from bag file
bag = rosbag.Bag('/home/haresh/PycharmProjects/visual_IKD/data/2022-02-06-12-53-20.bag')

# parse through bag file
joystick_data, odom_data, car_status = [], [], []
voltage_input, current_motor, current_input, charge_drawn, fault_code, temperature = [], [], [], [], [], []
odom_msg, car_status_msg, sensor_msg = None, None, None
time = []

# open a text file to write to:
# f = open('/home/haresh/PycharmProjects/visual_IKD/data/2022-02-06-11-39-09.txt', 'w')

for topic, msg, t in bag.read_messages(topics=['/vesc_drive', '/camera/odom/sample', '/car_status', '/sensors/core','/vectornav/IMU']):
	# if topic == '/vesc_drive':
	if topic == '/vectornav/IMU':
		# if odom_msg is None or car_status_msg is None: continue
		# if sensor_msg is None: continue
		# joystick_data.append([msg.twist.linear.x, msg.twist.angular.z])
		# odom_data.append([odom_msg.twist.twist.linear.x, odom_msg.twist.twist.angular.z])
		# car_status.append([car_status_msg.battery_voltage])
		# voltage_input.append([sensor_msg.state.voltage_input])
		# current_motor.append([sensor_msg.state.current_motor])
		# current_input.append([sensor_msg.state.current_input])
		# charge_drawn.append([sensor_msg.state.charge_drawn])
		# fault_code.append([sensor_msg.state.fault_code])
		temperature.append([sensor_msg.state.temperature_pcb])
		time.append(t.to_sec())

	elif topic=='/camera/odom/sample':
		odom_msg = msg

	elif topic=='/car_status':
		car_status_msg = msg

	elif topic=='/sensors/core':
		sensor_msg = msg

temperature = np.array(temperature)
time = np.array(time)-time[0]

np.savetxt('/home/haresh/PycharmProjects/visual_IKD/data/2022-02-06-12-53-20.csv', np.c_[time, temperature], delimiter=',')


# Plot data
joystick_data = np.array(joystick_data)
odom_data = np.array(odom_data)
car_status = np.array(car_status)
voltage_input = np.array(voltage_input)
current_motor = np.array(current_motor)
current_input = np.array(current_input)
charge_drawn = np.array(charge_drawn)
fault_code = np.array(fault_code)
temperature = np.array(temperature)
print(joystick_data.shape)
print(odom_data.shape)

# plt.subplot(2, 1, 1)
# plt.plot(np.arange(joystick_data[:].shape[0]), joystick_data[:, 0])
# plt.plot(np.arange(odom_data[:].shape[0]), odom_data[:, 0])
# # plt.plot(np.arange(current_input.shape[0]-5), current_input[:-5, 0])
# plt.ylabel('Linear Velocity')
# plt.legend(['Joystick', 'Realsense'])
# plt.subplot(2, 1, 2)
# plt.plot(np.arange(joystick_data[:].shape[0]), joystick_data[:, 1])
# plt.plot(np.arange(odom_data[:].shape[0]), odom_data[:, 1])
# plt.ylabel('Angular Velocity')
# plt.show()

# plt.figure()
# plt.plot(np.arange(car_status.shape[0]), car_status[:, 0])
# plt.show()

# plt.figure()
# plt.plot(np.arange(voltage_input.shape[0]), voltage_input[:, 0])
# plt.show()

# plt.figure()
# plt.plot(np.arange(current_motor.shape[0]), current_motor[:, 0])
# plt.show()

# plt.figure()
# plt.plot(np.arange(current_input.shape[0]), current_input[:, 0])
# plt.show()
#
# plt.figure()
# plt.plot(np.arange(charge_drawn.shape[0]), charge_drawn[:, 0])
# plt.show()
#
# plt.plot()
# plt.plot(np.arange(fault_code.shape[0]), fault_code[:, 0])
# plt.show()
# for i in range(fault_code.shape[0]):
# 	if fault_code[i] != 0:
# 		print(fault_code[i])
#

plt.figure(figsize=(20, 10))
plt.plot()
plt.plot(np.arange(temperature.shape[0]), temperature[:, 0])
plt.savefig('temp.png')
plt.show()
