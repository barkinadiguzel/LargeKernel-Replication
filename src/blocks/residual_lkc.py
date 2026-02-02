import torch.nn as nn
from .lkc_block import LKCBlock


class InvertedResidualLKC(nn.Module):
    def __init__(self, channels, kernel_sizes):
        super().__init__()
        self.block = LKCBlock(channels, kernel_sizes)

    def forward(self, x):
        return x + self.block(x)
