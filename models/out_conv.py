import torch
import torch.nn as nn

import models


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = models.DoubleConv(128, 64)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        y = self.conv(x)
        return y
