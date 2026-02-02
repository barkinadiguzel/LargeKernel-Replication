import torch.nn as nn
from layers.lkc_multi_branch import LKCMultiBranch


class LKCBlock(nn.Module):
    def __init__(self, channels, kernel_sizes):
        super().__init__()
        self.lkc = LKCMultiBranch(channels, kernel_sizes)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        return self.pw(self.lkc(x))
