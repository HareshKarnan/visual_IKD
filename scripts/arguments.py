import argparse

def get_args():
	parser = argparse.ArgumentParser(description='rosbag parser')
	parser.add_argument('--max_epochs', type=int, default=1000)
	parser.add_argument('--history_len', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--num_gpus', type=int, default=1)
	parser.add_argument('--hidden_size', type=int, default=32)
	parser.add_argument('--use_vision', action='store_true', default=False)
	parser.add_argument('--data_dir', type=str,
						default='/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/ahg_indoor_bags/')
	parser.add_argument('--train_dataset_names', type=str, nargs='+', default=['train1_data',
																			 'train2_data',
																			 'train3_data',
																			 'train4_data',
																			 'train5_data'])
	parser.add_argument('--val_dataset_names', type=str, nargs='+', default=['train6_data', 'train7_data', 'train8_data'])
	return parser.parse_args()
