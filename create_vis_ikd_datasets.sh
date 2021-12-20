roslaunch src/rosbag_sync_data_rerecorder/launch/rosbag_data_recorder.launch dataset:=train1
roslaunch src/rosbag_sync_data_rerecorder/launch/rosbag_data_recorder.launch dataset:=train2
roslaunch src/rosbag_sync_data_rerecorder/launch/rosbag_data_recorder.launch dataset:=train3
rm /robodata/kvsikand/visualIKD/train1_data/*
roslaunch src/rosbag_sync_data_rerecorder/launch/rosbag_data_recorder.launch dataset:=train1 data_base_dir:=/robodata/ut_alphatruck_logs/visualIKD/ out_base_dir:=/robodata/kvsikand/visualIKD/
rm /robodata/kvsikand/visualIKD/train2_data/*
roslaunch src/rosbag_sync_data_rerecorder/launch/rosbag_data_recorder.launch dataset:=train2 data_base_dir:=/robodata/ut_alphatruck_logs/visualIKD/ out_base_dir:=/robodata/kvsikand/visualIKD/
rm /robodata/kvsikand/visualIKD/train3_data/*
roslaunch src/rosbag_sync_data_rerecorder/launch/rosbag_data_recorder.launch dataset:=train3 data_base_dir:=/robodata/ut_alphatruck_logs/visualIKD/ out_base_dir:=/robodata/kvsikand/visualIKD/
rm /robodata/kvsikand/visualIKD/train4_data/*
roslaunch src/rosbag_sync_data_rerecorder/launch/rosbag_data_recorder.launch dataset:=train4 data_base_dir:=/robodata/ut_alphatruck_logs/visualIKD/ out_base_dir:=/robodata/kvsikand/visualIKD/