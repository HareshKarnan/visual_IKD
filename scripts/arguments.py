import argparse

def get_args():
	parser = argparse.ArgumentParser(description='rosbag parser')
	parser.add_argument('--max_epochs', type=int, default=1000)
	parser.add_argument('--history_len', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--num_gpus', type=int, default=1)
	parser.add_argument('--hidden_size', type=int, default=32)
	parser.add_argument('--use_vision', action='store_true', default=False)
	parser.add_argument('--use_simple_vision', action='store_true', default=False)
	parser.add_argument('--data_dir', type=str,
						default='/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/outdoor_bags/')
	parser.add_argument('--train_dataset_names', nargs='+', default=['train1_data', 'train3_data', 'train5_data',
																	 'train7_data', 'train9_data', 'train10_data', 'train11_data',
																	 'train13_data', 'train14_data', 'train15_data',
																	 'train17_data', 'train18_data', 'train20_data',
																	 'train22_data'])
	parser.add_argument('--val_dataset_names', nargs='+',
						default=['train2_data', 'train4_data',
								 'train6_data', 'train8_data',
								 'train12_data', 'train16_data',
								 'train21_data', 'train19_data'])
	# parser.add_argument('--train_dataset_names', nargs='+', default=['train1_data', 'train3_data', 'train5_data','train7_data', 'train9_data','train10_data', 'train11_data', 'train12_data'])
	# parser.add_argument('--val_dataset_names', nargs='+', default=['train2_data', 'train4_data', 'train6_data', 'train8_data'])

	return parser.parse_args()
