import argparse

def get_args():
	parser = argparse.ArgumentParser(description='rosbag parser')
	parser.add_argument('--max_epochs', type=int, default=1000)
	parser.add_argument('--history_len', type=int, default=4)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--hidden_size', type=int, default=32)
	parser.add_argument('--use_vision', action='store_true', default=False)
	parser.add_argument('--data_dir', type=str,
						default='/robodata/ut_alphatruck_logs/visualIKD/')
	parser.add_argument('--dataset_names', type=str, nargs='+', default=['train1'])
	return parser.parse_args()
