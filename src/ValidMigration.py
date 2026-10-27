"""
Stub for ConvOffset2D — used only for compute overhead measurement.
Replaces deformable conv with a standard conv of same param count.
"""
import torch.nn as nn


class ConvOffset2D(nn.Conv2d):
    """Lightweight stub: same parameter count as a 3×3 depthwise conv."""
    def __init__(self, in_channels):
        super().__init__(in_channels, in_channels, kernel_size=3, padding=1,
                         groups=in_channels, bias=False)

    def forward(self, x):
        return super().forward(x)
