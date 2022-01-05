import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='/robodata/kvsikand/visualIKD/')
parser.add_argument('--dataset_names', nargs='+', default=['train7'])

args = parser.parse_args()

triplets = []

def find_distant_patches(anchor_joystick, anchor_odom, data):
  anchor_joystick = np.asarray(anchor_joystick)
  anchor_odom = np.asarray(anchor_odom)
  data_len = len(data['odom'])
  distant_patch_indices = []
  for i in range(data_len):
    joystick = np.asarray(data['joystick'][i])
    odom = np.asarray(data['odom'][i])
    curr_odom = odom[:3]
    next_odom = odom[3:]

    # patches are distant if they have the *same* joystick and input odom but different odom responses...
    DIST_ODOM_THRESHOLD = 0.18
    IDENTICAL_ODOM_THRESHOLD  = 0.03
    if np.linalg.norm(anchor_joystick -joystick) < IDENTICAL_ODOM_THRESHOLD and np.linalg.norm(curr_odom - anchor_odom[:3]) < IDENTICAL_ODOM_THRESHOLD:
      if np.linalg.norm(next_odom - anchor_odom[3:]) > DIST_ODOM_THRESHOLD:
        distant_patch_indices.append(i)
      # else:
        # print('Skipping patch with same joystick and odom and same odom response')
        # print('RESPONSE DISTANCE', np.linalg.norm(next_odom - anchor_odom[3:]))
  return distant_patch_indices

for dataset in args.dataset_names:
  print('processing dataset', dataset)
  data_file = os.path.join(args.data_dir, dataset + '_data', 'data_1.pkl')
  data = pickle.load(open(data_file, 'rb'))
  data_len = len(data['odom'])
  segmented_patches = []
  for i in tqdm(range(data_len)):
    odom = data['odom'][i]
    joystick = data['joystick'][i]
    patches = data['patches'][i]

    negative = find_distant_patches(joystick, odom, data)
    if len(negative) > 0:
      segmented_patches.append((i, negative))

  # Creates segmented patches, where each entry is a pair of (positive patches, negative patches).
  # In each entry, all positive patches are similar to each other, but distant from each negative patch.
  print('Segmented Patches', len(segmented_patches))
  print('saving segmented patches to', os.path.join(args.data_dir, dataset + '_data', 'segmented_patches.pkl'))
  pickle.dump(segmented_patches, open(os.path.join(args.data_dir, dataset + '_data', 'segmented_patches.pkl'), 'wb'))

