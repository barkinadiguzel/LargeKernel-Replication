import torch.nn as nn
from blocks.inverted_residual_lkc import InvertedResidualLKC


class BackboneStub(nn.Module):
    def __init__(self, channels, kernel_sizes, depth=3):
        super().__init__()
        self.blocks = nn.Sequential(*[
            InvertedResidualLKC(channels, kernel_sizes)
            for _ in range(depth)
        ])

    def forward(self, x):
        return self.blocks(x)
