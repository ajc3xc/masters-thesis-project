# models/sebica.py

'''
Author: Adam Camerer via ChatGPT
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBICA(nn.Module):
    def __init__(self, in_channels):
        super(SEBICA, self).__init__()
        # Bidirectional Channel Attention
        self.channel_att = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Spatial Attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        b, c, h, w = x.size()
        x_perm = x.view(b, c, -1)
        channel_att = self.channel_att(x_perm)
        channel_att = channel_att.view(b, c, 1, 1)
        x = x * channel_att

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att

        return x