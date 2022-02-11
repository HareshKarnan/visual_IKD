#!/bin/bash

# run a loop from 1 to 10
for i in {11..13}
do
  echo "processing rosbag train$i.bag"
  rm /home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/ahg_indoor_bags/train$i_data/*
  roslaunch src/rosbag_sync_data_rerecorder/launch/rosbag_data_recorder.launch dataset:=train$i data_base_dir:=/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/ahg_indoor_bags out_base_dir:=/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/ahg_indoor_bags
done
