import matplotlib.pyplot as plt
import numpy as np
import glob
import rosbag
import yaml
from scipy.spatial.distance import directed_hausdorff

# read waypoints
# read the yaml file and load
with open('data/indoor_expt_data/waypoints.yaml') as f:
    waypoints = yaml.load(f, Loader=yaml.FullLoader)

waypoints = [[waypoints[val]["position"][0], waypoints[val]["position"][1]] for val in waypoints.keys()]
waypoints = np.array(waypoints)

rosbag = rosbag.Bag('data/indoor_expt_data/3_1.bag')
loc_x, loc_y, traced_trajectory = [], [], []
for topic, msg, t in rosbag.read_messages(topics=['/localization']):
    loc_x.append(msg.pose.x)
    loc_y.append(msg.pose.y)
    traced_trajectory.append([msg.pose.x, msg.pose.y])

traced_trajectory = np.array(traced_trajectory)

hd = max(directed_hausdorff(waypoints, traced_trajectory)[0], directed_hausdorff(traced_trajectory, waypoints)[0])
print('hausdorff distance : ', hd)
plt.xlim(min(waypoints[:, 0]) - 0.5, max(waypoints[:, 0]) + 0.5)
plt.ylim(min(waypoints[:, 1]) - 0.5, max(waypoints[:, 1]) + 0.5)
plt.plot(loc_x, loc_y, 'b')
plt.plot(waypoints[:, 0], waypoints[:, 1], 'r')
plt.show()