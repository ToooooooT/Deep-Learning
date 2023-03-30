import torch.nn as nn

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
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0), # (32, 1, 187)
            nn.Dropout(p=0.25)
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False), # (32, 1, 187)
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0), # (32, 1, 23)
            nn.Dropout(p=0.25),
            nn.Flatten()
        )
        self.l4 = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
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
        # input (channel, width, height) : (1, 2, 750)
        self.l1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5)), # (25, 2, 746)
            nn.Conv2d(25, 25, kernel_size=(self.C, 1)), # (25, 1, 746)
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)), # (25, 1, 373)
            nn.Dropout()
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5)), # (50, 1, 369)
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)), # (50, 1, 184)
            nn.Dropout()
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5)), # (100, 1, 180)
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)), # (100, 1, 90)
            nn.Dropout()
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5)), # (200, 1, 86)
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)), # (200, 1, 43)
            nn.Dropout(),
            nn.Flatten()
        )
        self.l5 = nn.Sequential(
            nn.Linear(200 * 1 * 43, 2), 
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        y = self.l5(x)
        return y