#!/bin/bash
# run a loop from 1 to 10
for i in {1..9}
do
  echo "processing rosbag train${i}.bag"
  rm "/robodata/kvsikand/visualIKD/train${i}_data/*"
  roslaunch src/rosbag_sync_data_rerecorder/launch/rosbag_data_recorder.launch dataset:="train$i" data_base_dir:=/robodata/ut_alphatruck_logs/visualIKD/ out_base_dir:=/robodata/kvsikand/visualIKD/
done