# models/MFS.py
'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org

MFS block created by adam camerer
'''

import torch
import torch.nn as nn
from models.GBC import GBC, BottConv
from models.DySample import DySample

class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
    def forward(self, x):
        return self.proj(x)

class WeightedFusion(nn.Module):
    """Learnable fusion of multi-scale features"""
    def __init__(self, num_scales):
        super().__init__()
        # one weight per feature map
        self.weights = nn.Parameter(torch.ones(num_scales))
    def forward(self, features):
        w = torch.softmax(self.weights, dim=0)
        # weighted sum of features
        return sum(f * w[i] for i, f in enumerate(features))

class MFS(nn.Module):
    def __init__(self, embedding_dim):
        super(MFS, self).__init__()
        self.embedding_dim = embedding_dim

        # project each C_i → embedding_dim
        self.linear_c4 = MLP(input_dim=128, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=64,  embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=32,  embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=16,  embed_dim=embedding_dim)

        # dynamic upsampling
        self.DySample_C_8 = DySample(embedding_dim, scale=8)
        self.DySample_C_4 = DySample(embedding_dim, scale=4)
        self.DySample_C_2 = DySample(embedding_dim, scale=2)

        # **Replace** static concat + GBC on 4x channels
        self.fusion   = WeightedFusion(num_scales=4)
        self.GBC_C    = GBC(embedding_dim)     # now only embedding_dim channels
        self.linear_fuse = BottConv(embedding_dim, embedding_dim,
                                    embedding_dim//8, kernel_size=1)

        # final prediction head
        self.dropout     = nn.Dropout(p=0.1)
        self.linear_pred = BottConv(embedding_dim, 1, 1, kernel_size=1)
        self.linear_pred_1 = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, inputs):
        c4, c3, c2, c1 = inputs
        b, c, h, w = c4.shape
        out_c4 = self.linear_c4(c4.reshape(b,c,h*w).permute(0,2,1))\
                   .permute(0,2,1).reshape(b, self.embedding_dim, h, w)
        out_c4 = self.DySample_C_8(out_c4)

        b, c, h, w = c3.shape
        out_c3 = self.linear_c3(c3.reshape(b,c,h*w).permute(0,2,1))\
                   .permute(0,2,1).reshape(b, self.embedding_dim, h, w)
        out_c3 = self.DySample_C_4(out_c3)

        b, c, h, w = c2.shape
        out_c2 = self.linear_c2(c2.reshape(b,c,h*w).permute(0,2,1))\
                   .permute(0,2,1).reshape(b, self.embedding_dim, h, w)
        out_c2 = self.DySample_C_2(out_c2)

        b, c, h, w = c1.shape
        out_c1 = self.linear_c1(c1.reshape(b,c,h*w).permute(0,2,1))\
                   .permute(0,2,1).reshape(b, self.embedding_dim, h, w)

        # **Use** learnable fusion instead of concat
        fused = self.fusion([out_c4, out_c3, out_c2, out_c1])
        out_c = self.GBC_C(fused)
        out_c = self.linear_fuse(out_c)

        out_c = self.dropout(out_c)
        x = self.linear_pred_1(self.linear_pred(out_c))
        return x
