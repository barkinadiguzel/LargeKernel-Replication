import torch.nn as nn


class DWConv(nn.Module):
    def __init__(self, channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=False
        )

    def forward(self, x):
        return self.conv(x)
