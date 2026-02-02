import torch.nn as nn
from .dwconv import DWConv


class LKCMultiBranch(nn.Module):
    def __init__(self, channels, kernel_sizes):
        super().__init__()
        self.branches = nn.ModuleList([
            DWConv(channels, k, k // 2) for k in kernel_sizes
        ])
        self.id = nn.Identity()

    def forward(self, x):
        out = self.id(x)
        for b in self.branches:
            out = out + b(x)
        return out
