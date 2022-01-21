import numpy as np
import torch
import pickle
import cv2
from scipy.spatial.transform import Rotation as R
from pytorch3d import transforms as PT
import time
import torchgeometry as tgm

C_i = np.array(
	[622.0649233612024, 0.0, 633.1717569157071, 0.0, 619.7990184421728, 368.0688607187958, 0.0, 0.0, 1.0]).reshape(
	(3, 3))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
R_cam_imu = R.from_euler("xyz", [90, -90, 0], degrees=True)
R2 = R.from_euler("xyz", [0, 0, -90], degrees=True).as_matrix()
n = np.array([0, 0, 1]).reshape((3, 1))


def homography_camera_displacement(R1, R2, t1, t2, n1):
	R12 = R2 @ R1.T
	t12 = R2 @ (- R1.T @ t1) + t2
	# d is distance from plane to t1.
	d = np.linalg.norm(n1.dot(t1.T))

	H12 = R12 - ((t12 @ n1.T) / d)
	H12 /= H12[2, 2]
	return H12


def camera_imu_homography_numpy(imu, image):
	imu = np.array([imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w])

	R_imu_world = R.from_quat(imu).as_euler('xyz', degrees=False)
	R_imu_world[0], R_imu_world[1] = -R_imu_world[0], R_imu_world[1]
	R_imu_world[2] = 0.
	R_imu_world = R.from_euler('xyz', R_imu_world, degrees=False)


	R1 = R_cam_imu * R_imu_world
	R1 = R1.as_matrix()

	t1 = R1 @ np.array([0., 0., 0.5]).reshape((3, 1))
	t2 = R2 @ np.array([-2.5, -0., 6.0]).reshape((3, 1))
	n1 = R1 @ n

	H12 = homography_camera_displacement(R1, R2, t1, t2, n1)
	homography_matrix = C_i @ H12 @ np.linalg.inv(C_i)
	homography_matrix /= homography_matrix[2, 2]

	start_time = time.time()
	output = cv2.warpPerspective(image, homography_matrix, (1280, 720))
	# flip output horizontally
	output_cpu = cv2.flip(output, 1)
	# print("--- %f seconds CPU ---" % float((time.time() - start_time)/100.))
	cpu_time = float((time.time() - start_time)/100.)

	start_time = time.time()
	image_torch = torch.from_numpy(image.astype(np.float32)/255.0).to(device).reshape((1, 3, 720, 1280)).cuda()
	homography_matrix_torch = torch.from_numpy(homography_matrix.astype(np.float32)).to(device).reshape((1, 3, 3)).cuda()
	warper = tgm.HomographyWarper(720, 1280)
	output_torch = warper(image_torch.float(), homography_matrix_torch.float())
	# print("--- %f seconds GPU ---" % float((time.time() - start_time)/100.))
	gpu_time = float((time.time() - start_time)/100.)

	output_torch = output_torch.cpu().detach().numpy()*255.0
	output_torch = output_torch.astype(np.uint8)
	output_gpu = cv2.flip(output, 1)

	print('output gpu shape : ', output_gpu.shape)
	print('output gpu dtype : ', output_gpu.dtype)

	return output_cpu, output_gpu, cpu_time, gpu_time


if __name__ == '__main__':
	# load the pickle files
	image = pickle.load(open('tmp/image_msg.pkl', 'rb'))
	imu = pickle.load(open('tmp/imu_msg.pkl', 'rb'))

	# convert imu and image to numpy arrays
	img = np.fromstring(image.data, np.uint8)
	image = cv2.imdecode(img, cv2.IMREAD_COLOR)

	cpu_times, gpu_times = [], []
	for _ in range(100):
		bevimage_cpu, bevimage_gpu, cpu_time, gpu_time = camera_imu_homography_numpy(imu, image)
		cpu_times.append(cpu_time)
		gpu_times.append(gpu_time)

	print("Average CPU time: %f" % np.mean(cpu_times))
	print("Average GPU time: %f" % np.mean(gpu_times))

	cv2.imshow('bevimage CPU', bevimage_cpu)
	cv2.imshow('bevimage GPU', bevimage_gpu)
	cv2.waitKey(0)



