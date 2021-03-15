import torch
import torch.nn as nn

import models


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = models.DoubleConv(in_channels, out_channels, mid_channels)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        y = self.up(x)
        return y
