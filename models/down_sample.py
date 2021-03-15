import torch.nn as nn

import models


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = models.DoubleConv(in_channels, out_channels)

    def forward(self, x):
        y = self.down(x)
        y = self.conv(y)
        return y
