import torch
from torch import nn
import torch.nn.functional as F

class L2Normalize(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1) # L2 normalize

class VisualIKDNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(VisualIKDNet, self).__init__()
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32), nn.PReLU(), # 31x31
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.PReLU(), # 15x15
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128), nn.PReLU(), # 7x7
            nn.Flatten(),
            nn.Linear(7*7*128, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 16), nn.Tanh(),
        )

        self.trunk = nn.Sequential(
            nn.Linear(input_size + 16, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, non_image, image):
        visual_embedding = self.visual_encoder(image)
        output = self.trunk(torch.cat((non_image, visual_embedding), dim=1))
        return output

class SimpleIKDNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(SimpleIKDNet, self).__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, non_image):
        return self.trunk(non_image)