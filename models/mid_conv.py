import torch.nn as nn

import models


class MidConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.down = nn.MaxPool2d(2)
        self.double_conv = models.DoubleConv(in_channels, out_channels, mid_channels)

    def forward(self, x):
        y = self.down(x)
        y = self.double_conv(y)
        y = self.up(y)
        return y
