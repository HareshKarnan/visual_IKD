#!/bin/bash

# run a loop from 1 to 10
for i in {13..28}
do
  echo "processing rosbag train$i.bag"
  rm /home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/outdoor_bags/train$i_data/*
  roslaunch src/rosbag_sync_data_rerecorder/launch/rosbag_data_recorder.launch dataset:=train$i data_base_dir:=/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/outdoor_bags out_base_dir:=/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/outdoor_bags
done
