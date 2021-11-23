### config file

Add the specific robot config in the config file at `config/robot_name.yaml`

### Generating the dataset `*_data.pkl` file
Set the rosbag that you want to process and pickle into a dataset in this roslaunch file : `src/rosbag_sync_data_rerecorder/launch/rosbag_data_recorder.launch`

This launch file will generate a dataset file in the `data/` folder. 
Point to this dataset file in `scripts/train.py` to train the visual IKD model.