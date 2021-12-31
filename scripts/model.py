import torch
from torch import nn
import torch.nn.functional as F
from resnet8_model import ResNet8

class L2Normalize(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1) # L2 normalize

class VisualIKDNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(VisualIKDNet, self).__init__()
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2), nn.ReLU(), # 30 x 30
            nn.MaxPool2d(kernel_size=2, stride=2), # 15 x 15
            nn.Conv2d(6, 12, kernel_size=3, stride=2), nn.ReLU(), # 7 x 7
            nn.MaxPool2d(kernel_size=2, stride=2), # 3 x 3
            nn.Flatten(),
            nn.Linear(3*3*12, 16)
        )

        # self.visual_encoder = ResNet8(output_emb_size=16)

        self.imu_net = nn.Sequential(
            nn.Linear(200 * 3 + 60 * 3, 128), nn.BatchNorm1d(128), nn.PReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.PReLU(),
            nn.Linear(64, 16),
            nn.Dropout(p=0.1)
        )

        self.ikdmodel = nn.Sequential(
            nn.Linear(input_size - (200 * 3 + 60 * 3) + 16 + 16, hidden_size), nn.BatchNorm1d(hidden_size), nn.PReLU(),
            nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.PReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, accel, gyro, odom, image):
        visual_embedding = self.visual_encoder(image)
        accel_gyro = torch.cat((accel, gyro), dim=1)
        imu_embedding = self.imu_net(accel_gyro)
        return self.ikdmodel(torch.cat((odom, imu_embedding, visual_embedding), dim=1))

class SimpleIKDNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(SimpleIKDNet, self).__init__()
        self.imu_net = nn.Sequential(
            nn.Linear(200*3 + 60 * 3, 128), nn.BatchNorm1d(128), nn.PReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.PReLU(),
            nn.Linear(64, 16),
            nn.Dropout(p=0.1)
        )
        self.ikdmodel = nn.Sequential(
            nn.Linear(input_size - (200*3 + 60 * 3) + 16, hidden_size), nn.BatchNorm1d(hidden_size), nn.PReLU(),
            nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.PReLU(),
            nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.PReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, accel, gyro, odom):
        accel_gyro = torch.cat((accel, gyro), dim=1)
        imu_embedding = self.imu_net(accel_gyro)
        return self.ikdmodel(torch.cat((odom, imu_embedding), dim=1))
