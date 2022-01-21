import matplotlib.pyplot as plt
import numpy as np
from scripts.visualize_trial_data import load_waypoints
from scipy.signal import savgol_filter

def resample_waypoints(waypoints, alpha=2):
	return waypoints[::alpha]

if __name__ == '__main__':
	waypoints = load_waypoints('waypoints.yaml')
	waypoints = np.asarray(waypoints[:-35])

	waypoints[:, 0] = savgol_filter(waypoints[:, 0], 5, 2)
	waypoints[:, 1] = savgol_filter(waypoints[:, 1], 5, 2)

	print(waypoints)
	waypoints = resample_waypoints(waypoints, alpha=5)

	# Plot the waypoints
	plt.figure()
	plt.plot(waypoints[:,0], waypoints[:,1], 'o')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Waypoints')
	plt.show()