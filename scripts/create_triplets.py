import argparse
import pickle
import os
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='/robodata/ut_alphatruck_logs/visualIKD/')
parser.add_argument('--dataset_names', nargs='+', default=['train1', 'train3', 'train4'])

args = parser.parse_args()

triplets = []

def find_distant_patch(anchor_joystick, anchor_odom, data):
  anchor_joystick = np.asarray(anchor_joystick)
  anchor_odom = np.asarray(anchor_odom)
  data_len = len(data['odom'])
  for i in range(data_len):
    joystick = np.asarray(data['joystick'][i])
    odom = np.asarray(data['odom'][i])

    # patches are distant if they have the *same* joystick but different odom responses...I suppose
    # Right now we just have current odom, not the odom response in this particular datapoint...

for dataset in args.dataset_names:
  data_file = os.path.join(args.data_dir, dataset, 'data_1.pkl')
  data = pickle.load(open(data_file, 'rb'))
  data_len = len(data['odom'])
  for i in range(data_len):
    odom = data['odom'][i]
    joystick = data['joystick'][i]
    patches = data['patches'][i]

    anchor_idx, sim_idx = np.random.randint(0, patches.shape[0], size=2)
    anchor = patches[anchor_idx]
    similar = patches[sim_idx]

    negative = find_distant_patch(joystick, odom)
    


