import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):

    def __init__(self, n_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)

        self.conv2 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, ip):
        x = self.bn1(self.conv1(ip))
        x = F.leaky_relu(x, 0.1)

        x = self.bn2(self.conv2(x))
        x = F.leaky_relu(x, 0.1)

        return x + ip


class ResNet(nn.Module):

    def __init__(self, device, n_channels=128, n_res_blocks=2, ip_channels=3):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(ip_channels, n_channels, 1)

        self.res_blocks = [ResNetBlock(n_channels) for _ in range(n_res_blocks)]
        self.res_blocks = [x.to(device) for x in self.res_blocks]

        self.conv2 = nn.Conv2d(n_channels, n_channels, 1)
        self.conv3 = nn.Conv2d(n_channels, n_channels, 1)
        self.conv4 = nn.Conv2d(n_channels, ip_channels, 1)

    def forward(self, x):
        avg = torch.mean(x, axis=[2, 3], keepdim=True)  # Check axis
        x = x - avg
        x = self.conv1(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.conv3(x)
        x = F.leaky_relu(x, 0.1)
        x = self.conv4(x)

        x = x + avg
        return x
