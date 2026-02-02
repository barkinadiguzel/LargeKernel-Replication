import torch.nn as nn
from kernels.reparam_rules import fuse_kernels


class LKCReparam(nn.Module):
    def __init__(self, channels, kernel_sizes):
        super().__init__()
        self.target_size = max(kernel_sizes)
        self.conv = nn.Conv2d(
            channels,
            channels,
            self.target_size,
            padding=self.target_size // 2,
            groups=channels,
            bias=False
        )

    def load_from_multi_branch(self, branches):
        kernels = [b.conv.weight.data for b in branches]
        fused = fuse_kernels(kernels, self.target_size)
        self.conv.weight.data.copy_(fused)

    def forward(self, x):
        return self.conv(x)
