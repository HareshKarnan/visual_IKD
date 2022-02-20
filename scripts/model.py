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
        # self.visual_encoder = nn.Sequential(
        #      nn.Conv2d(3, 3, kernel_size=3, stride=2),
        #      nn.BatchNorm2d(3), nn.PReLU(), # 31x31
        #      nn.Conv2d(3, 6, kernel_size=3, stride=2),
        #      nn.BatchNorm2d(6), nn.PReLU(), # 15x15
        #      nn.Conv2d(6, 8, kernel_size=3, stride=2),
        #      nn.BatchNorm2d(8), nn.PReLU(), # 7x7
        #      nn.Flatten(),
        #      nn.Linear(7*7*8, hidden_size), nn.PReLU(),
        #      nn.Linear(hidden_size, 16)
        # )

        self.visual_encoder = nn.Sequential(
             nn.Conv2d(3, 16, kernel_size=3, stride=2), nn.BatchNorm2d(16), nn.PReLU(), # 31x31
             nn.MaxPool2d(kernel_size=3, stride=2), # 15x15
             nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.BatchNorm2d(32), nn.PReLU(), # 7x7
             nn.MaxPool2d(kernel_size=3, stride=2), # 3x3
             nn.Flatten(),
             nn.Linear(3*3*32, 8)
        )
        
        self.imu_net = nn.Sequential(
            nn.Linear(200 * 3 + 60 * 3, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 4),
        )
        self.imu_skip = nn.Linear(200 * 3 + 60 * 3, 4)

        self.ikdmodel = nn.Sequential(
            nn.Linear(2 + 4 + 8, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        self.ikdmodel_skip = nn.Linear(2 + 4 + 8, output_size)

    def forward(self, accel, gyro, odom, image, patch_observed):
        visual_embedding = self.visual_encoder(image)
        unobserved_indices = torch.nonzero(torch.logical_not(patch_observed)).squeeze()
        visual_embedding[unobserved_indices] = torch.zeros((1, 8)).cuda()
        imu_embedding = self.imu_net(torch.cat((accel, gyro), dim=1)) + self.imu_skip(torch.cat((accel, gyro), dim=1))
        return self.ikdmodel(torch.cat((odom, imu_embedding, visual_embedding), dim=1)) + self.ikdmodel_skip(torch.cat((odom, imu_embedding, visual_embedding), dim=1))

class SimpleIKDNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(SimpleIKDNet, self).__init__()
        self.imu_net = nn.Sequential(
            nn.Linear(200*3 + 60 * 3, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 4),
        )
        self.imu_skip = nn.Linear(200*3 + 60 * 3, 4)

        self.ikdmodel = nn.Sequential(
            nn.Linear(2 + 4, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        self.ikdmodel_skip = nn.Linear(2 + 4, output_size)

        self.full_skip = nn.Linear(200*3 + 60 * 3, output_size)

    def forward(self, accel, gyro, odom):
        accel_gyro = torch.cat((accel, gyro), dim=1)
        imu_embedding = self.imu_net(accel_gyro) + self.imu_skip(accel_gyro)
        return self.ikdmodel(torch.cat((odom, imu_embedding), dim=1)) + self.ikdmodel_skip(torch.cat((odom, imu_embedding), dim=1)) + self.full_skip(accel_gyro)
