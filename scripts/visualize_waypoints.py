import copy
import matplotlib.pyplot as plt
import numpy as np
from scripts.visualize_trial_data import load_waypoints_position
from scipy.signal import savgol_filter

def resample_waypoints(waypoints, alpha=2):
	return waypoints[::alpha]

if __name__ == '__main__':
	waypoints = load_waypoints_position('waypoints.yaml')
	# waypoints = np.asarray(waypoints[:-35])

	waypoints_filtered = copy.deepcopy(waypoints)

	waypoints_filtered[:, 0] = savgol_filter(waypoints[:, 0], 99, 3)
	waypoints_filtered[:, 1] = savgol_filter(waypoints[:, 1], 99, 3)

	waypoints = resample_waypoints(waypoints, alpha=10)
	waypoints_filtered = resample_waypoints(waypoints_filtered, alpha=10)

	# Plot the waypoints
	plt.figure()
	plt.plot(waypoints[:,0], waypoints[:,1], 'ob')
	plt.plot(waypoints_filtered[:,0], waypoints_filtered[:,1], 'og')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Waypoints')
	plt.show()