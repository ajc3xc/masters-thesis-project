# models/sfa.py

'''
Author: Adam Camerer via ChatGPT
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class SFA(nn.Module):
    def __init__(self, in_channels, groups=4):
        super(SFA, self).__init__()
        self.groups = groups
        self.group_channels = in_channels // groups

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(self.group_channels, self.group_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.group_channels, self.group_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial Attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size * self.groups, self.group_channels, height, width)

        # Channel Attention
        avg = self.avg_pool(x)
        channel_att = self.channel_fc(avg)
        x = x * channel_att

        x = x.view(batch_size, channels, height, width)

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att

        return x