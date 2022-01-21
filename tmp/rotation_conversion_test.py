import numpy as np
from scipy.spatial.transform import Rotation as R
from pytorch3d import transforms as PT
import torch

# generate random number from 0-90
euler = np.random.uniform(0, 90, 3)

rot = R.from_euler('xyz', euler, degrees=False).as_matrix()
print(rot)

rot = PT.euler_angles_to_matrix(torch.tensor(-euler), 'XYZ').T
print(rot)
