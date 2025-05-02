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
        self.channel_reduce = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.spatial_att = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # Channel Attention
        x_perm = x.flatten(2)                  # [B, C, H*W]
        channel_att = self.channel_reduce(x_perm).mean(-1).view(b, c, 1, 1)
        x = x * self.sigmoid(channel_att)

        # Spatial Attention
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial_att(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        return x