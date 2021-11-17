import torch
from torch import nn

class VisualIKDNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(VisualIKDNet, self).__init__()
        self.flatten = nn.Flatten()

        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2), nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2), nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2), nn.PReLU(),
            self.flatten,
            nn.Linear(256*3*3, 128), nn.PReLU(),
            nn.Linear(128, 16), nn.Tanh()
        )

        self.trunk = nn.Sequential(
            nn.Linear(input_size + 16, hidden_size), nn.PReLU(),
            nn.Linear(hidden_size, hidden_size), nn.PReLU(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, non_image, image):
        visual_embedding = self.visual_encoder(image)
        output = self.trunk(torch.cat((non_image, visual_embedding), dim=1))
        return output