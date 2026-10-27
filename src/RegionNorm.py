"""
Stub for RCNModule — used only for compute overhead measurement.
Replaces Regional Composite Normalization with InstanceNorm2d.
"""
import torch.nn as nn


class RCNModule(nn.Module):
    """Lightweight stub: replaces RCN with InstanceNorm."""
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=True)

    def forward(self, x, mask=None):
        return self.norm(x)
