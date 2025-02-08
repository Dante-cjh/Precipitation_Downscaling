import torch.nn as nn
import torch

class SRCNN(nn.Module):
    def __init__(self, input_channels=2):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x