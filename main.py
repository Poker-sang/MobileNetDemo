import torch
import torch.nn as nn


class Conv2dDw(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(Conv2dDw, self).__init__(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class MobileNet(nn.Module):
    loss_function = nn.CrossEntropyLoss()

    def __init__(self):
        super(MobileNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # 输入大小 3 * 224 * 224
        self.layers = nn.Sequential(
            Conv2dDw(32, 64),
            Conv2dDw(64, 128, stride=2),
            Conv2dDw(128, 128),
            Conv2dDw(128, 256, stride=2),
            Conv2dDw(256, 256),
            Conv2dDw(256, 512, stride=2),
            Conv2dDw(512, 512),
            Conv2dDw(512, 512),
            Conv2dDw(512, 512),
            Conv2dDw(512, 512),
            Conv2dDw(512, 512),
            Conv2dDw(512, 1024, stride=2),
            Conv2dDw(1024, 1024),

            nn.AvgPool2d(7)
        )
        self.linear = nn.Linear(1024, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        out = self.conv(x)
        out = self.layers(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out
