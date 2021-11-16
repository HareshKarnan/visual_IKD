import argparse
import torch
from scripts.train import IKDDataModule, IKDModel
parser = argparse.ArgumentParser(description='rosbag parser')
parser.add_argument('--rosbag_path', type=str, default='data/ahg_road.bag')
parser.add_argument('--history_len', type=int, default=20)
parser.add_argument('--frequency', type=int, default=20)
parser.add_argument('--max_time', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model_path', type=str, default='models/model.pth')
parser.add_argument('--out_path', type=str, default='models/model.jit')
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"

model = IKDModel(input_size=6 + 3*args.history_len, output_size=2*args.history_len, hidden_size=256)

model = model.to(device=device)
model.load_state_dict(torch.load(args.model_path)["state_dict"])

topics_to_read = [
  '/camera/odom/sample',
  '/joystick',
  '/camera/accel/sample',
  '/camera/gyro/sample',
  '/webcam/image_raw/compressed'
]

keys = ['rgb', 'odom', 'accel', 'gyro', 'joystick']

dm = IKDDataModule(rosbag_path=args.rosbag_path,
                    frequency=args.frequency,
                    max_time=args.max_time,
                    config_path=args.config_path,
                    topics_to_read=topics_to_read,
                    keys=keys, batch_size=args.batch_size,
                    history_len=args.history_len)
                    
datum = next(iter(dm.val_dataloader()))

print("INPUT", datum.shape)

torchscript_model = torch.jit.trace(model, datum)

torchscript_model.save(args.out_path)

test_model = torch.jit.load(args.out_path)

test_outputs = test_model(datum)

print(test_outputs)