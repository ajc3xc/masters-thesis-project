# models/GBC.py
'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org

ECABlock created by adam camerer
'''

import torch
import torch.nn as nn

class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise   = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x

def get_norm_layer(norm_type, channels, num_groups):
    if norm_type == 'GN':
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        return nn.InstanceNorm3d(channels)

class ECABlock(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channels, k_size=3):
        super(ECABlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W) → squeeze to (B, 1, C)
        y = self.pool(x).squeeze(-1).permute(0, 2, 1)
        y = self.conv(y).permute(0, 2, 1).unsqueeze(-1)
        return x * self.sigmoid(y)

class GBC(nn.Module):
    def __init__(self, in_channels, norm_type='GN'):
        super(GBC, self).__init__()
        # same blocks as before
        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels//8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels//16),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels//8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels//16),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels//8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, in_channels//16),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels//8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, 16),
            nn.ReLU()
        )
        # **Add** this
        self.eca = ECABlock(in_channels)

    def forward(self, x):
        residual = x
        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.block3(x)
        x  = x1 * x2
        x  = self.block4(x)
        # **Add** this line to apply channel attention
        x  = self.eca(x)
        return x + residual
