import torch.nn as nn

import models


class UNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.in_conv = models.DoubleConv(args.channels, 64)
        self.down = models.DownSample(64, 128)
        self.mid = models.MidConv(128, 128, 256)
        self.up = models.UpSample(256, 64, 128)
        self.out_conv = models.OutConv(64, args.channels)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down(x1)
        x3 = self.mid(x2)
        y = self.up(x3, x2)
        y = self.out_conv(y, x1)
        return y
