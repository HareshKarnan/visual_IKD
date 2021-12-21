import argparse
import pickle
import os
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='/robodata/kvsikand/visualIKD/')
parser.add_argument('--dataset_names', nargs='+', default=['train1', 'train3', 'train4'])

args = parser.parse_args()

triplets = []

def find_distant_patches(anchor_joystick, anchor_odom, data):
  anchor_joystick = np.asarray(anchor_joystick)
  anchor_odom = np.asarray(anchor_odom)
  data_len = len(data['odom'])
  distant_patches = []
  for i in range(data_len):
    joystick = np.asarray(data['joystick'][i])
    odom = np.asarray(data['odom'][i])
    curr_odom = odom[:3]
    next_odom = odom[3:]

    # patches are distant if they have the *same* joystick and input odom but different odom responses...
    DIST_ODOM_THRESHOLD = 0.2
    IDENTICAL_ODOM_THRESHOLD  = 0.02
    if np.allclose(anchor_joystick, joystick, 1e-2) and  np.linalg.norm(curr_odom - anchor_odom[:3]) < IDENTICAL_ODOM_THRESHOLD:
      if np.linalg.norm(next_odom - anchor_odom[3:]) > DIST_ODOM_THRESHOLD:
        print('FOUND NEGATIVE SAMPLE', i, anchor_odom, odom, anchor_joystick, joystick)
        distant_patches.extend(data['patches'][i])
      # else:
        # print('Skipping patch with same joystick and odom and same odom response')
        # print('RESPONSE DISTANCE', np.linalg.norm(next_odom - anchor_odom[3:]))

for dataset in args.dataset_names:
  data_file = os.path.join(args.data_dir, dataset + '_data', 'data_1.pkl')
  data = pickle.load(open(data_file, 'rb'))
  data_len = len(data['odom'])
  for i in range(data_len):
    odom = data['odom'][i]
    joystick = data['joystick'][i]
    patches = data['patches'][i]

    anchor_idx, sim_idx = np.random.randint(0, len(patches), size=2)
    anchor = patches[anchor_idx]
    similar = patches[sim_idx]

    negative = find_distant_patches(joystick, odom, data)
    


