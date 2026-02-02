import torch.nn as nn
from .backbone_stub import BackboneStub


class LKCNetStub(nn.Module):
    def __init__(self, channels=64, kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.backbone = BackboneStub(channels, kernel_sizes)

    def forward(self, x):
        return self.backbone(x)
