import argparse
import torch
from scripts.train import IKDModel
from scripts.dataset import MyDataset
import pickle
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='rosbag parser')
parser.add_argument('--history_len', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--model_path', type=str, default='/home/haresh/PycharmProjects/visual_IKD/models/with_all_augmentations.ckpt')
parser.add_argument('--out_path', type=str, default='models/model.jit')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = IKDModel.load_from_checkpoint(args.model_path)
model = model.to(device=device)

data = pickle.load(open('/home/haresh/PycharmProjects/visual_IKD/src/rosbag_sync_data_rerecorder/data/ahg_indoor_bags/test1_data.pkl', 'rb'))

dataset = MyDataset(data, args.history_len)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                        drop_last=not (len(dataset) % args.batch_size == 0.0))
                    
datum = next(iter(dataloader))
odom, joystick, accel, gyro, bevimage = datum

non_visual_input = torch.cat((odom, accel, gyro), dim=1).to(device=device)
bevimage = bevimage.permute(0, 3, 1, 2).to(device=device)

torchscript_model = torch.jit.trace(model, (non_visual_input.float(), bevimage.float()))

torchscript_model.save(args.out_path)

test_model = torch.jit.load(args.out_path).to(device=device)

test_outputs = test_model(non_visual_input.float(), bevimage.float())

print(test_outputs)
print(test_outputs.shape)