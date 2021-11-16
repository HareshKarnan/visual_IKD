import argparse
import torch
from scripts.train import IKDDataModule, IKDModel
parser = argparse.ArgumentParser(description='rosbag parser')
parser.add_argument('--rosbag_path', type=str, default='data/ahg_road.bag')
parser.add_argument('--history_len', type=int, default=20)
parser.add_argument('--frequency', type=int, default=20)
parser.add_argument('--max_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--config_path', type=str, default="config/alphatruck.yaml")
parser.add_argument('--model_path', type=str, default='/home/haresh/PycharmProjects/visual_IKD/models/15-11-2021-23-43-30.ckpt')
parser.add_argument('--out_path', type=str, default='models/model.jit')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = IKDModel(input_size=6 + 3*args.history_len, output_size=2*args.history_len, hidden_size=256)

model.load_state_dict(torch.load(args.model_path)["state_dict"])
model = model.to(device=device)

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
odom, joystick, accel, gyro, bevimage = datum

non_visual_input = torch.cat((odom, accel, gyro), dim=1).to(device=device)
bevimage = bevimage.permute(0, 3, 1, 2).to(device=device)

torchscript_model = torch.jit.trace(model, (non_visual_input.float(), bevimage.float()))

torchscript_model.save(args.out_path)

test_model = torch.jit.load(args.out_path).to(device=device)

test_outputs = test_model(non_visual_input.float(), bevimage.float())

print(test_outputs)
print(test_outputs.shape)