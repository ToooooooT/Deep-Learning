import torch
import torch.nn as nn
import os
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False), # (16, 2, 750)
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.l2 = nn.Sequential( # groups: 1 in_channel -> 2 out_channel => parameters: 1(in_channel) x 2(out_channel) x 2(kernel_size) x 16 (groups) = 64
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),  # (32, 1, 750)
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0), # (32, 1, 187)
            nn.Dropout(p=0.5)
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False), # (64, 1, 187)
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0), # (64, 1, 23)
            nn.Dropout(p=0.5),
            nn.Flatten()
        )
        self.l4 = nn.Sequential(
            nn.Linear(in_features=1472, out_features=2, bias=True)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        y = self.l4(x)
        return y

class DeepConvNet(nn.Module):
    def __init__(self, C, T) -> None:
        super().__init__()
        self.C = C # 2
        self.T = T
        channels = [25, 25, 50, 50, 50]
        # input (channel, width, height) : (1, 2, 750)
        self.l1 = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=(1, 5)), # (25, 2, 746)
            nn.Conv2d(channels[0], channels[1], kernel_size=(self.C, 1)), # (25, 1, 746)
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)), # (25, 1, 373)
            nn.Dropout()
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=(1, 5)), # (50, 1, 369)
            nn.BatchNorm2d(channels[2], eps=1e-05, momentum=0.1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)), # (50, 1, 184)
            nn.Dropout()
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=(1, 5)), # (100, 1, 180)
            nn.BatchNorm2d(channels[3], eps=1e-05, momentum=0.1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)), # (100, 1, 90)
            nn.Dropout()
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], kernel_size=(1, 5)), # (200, 1, 86)
            nn.BatchNorm2d(channels[4], eps=1e-05, momentum=0.1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)), # (200, 1, 43)
            nn.Dropout(),
            nn.Flatten()
        )
        self.l5 = nn.Sequential(
            nn.Linear(channels[4] * 1 * 43, 2), 
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        y = self.l5(x)
        return y


class ShallowConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = [32, 32]
        self.l1 = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=(1, 13)),
            nn.Conv2d(channels[0], channels[1], kernel_size=(2, 1)),
            nn.BatchNorm2d(channels[1])
        )
        self.pool = nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7))
        self.l2 = nn.Sequential(
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(channels[1] * 101, 2)
        )

    def forward(self, x):
        x = self.l1(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(x)
        y = self.l2(x)
        return y