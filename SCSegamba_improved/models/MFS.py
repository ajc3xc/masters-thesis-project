'''
Author: Hui Liu
Modified by: Adam Camerer via ChatGPT
'''

from typing import List

import torch
import torch.nn as nn
from models.GBC import GBC, BottConv
from models.ECA import ECA
from models.SFA import SFA
from models.SEBICA import SEBICA
from models.DySample import DySample
from models.WF import WeightedFusion
from models.DWF import DynamicWeightedFusion

class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
    def forward(self, x):
        return self.proj(x)

class MFS(nn.Module):
    def __init__(self, embedding_dim, fusion_mode=None, attention_type=None):
        super(MFS, self).__init__()
        self.embedding_dim = embedding_dim

        self.linear_c4 = MLP(input_dim=128, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=64,  embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=32,  embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=16,  embed_dim=embedding_dim)

        self.DySample_C_8 = DySample(embedding_dim, scale=8)
        self.DySample_C_4 = DySample(embedding_dim, scale=4)
        self.DySample_C_2 = DySample(embedding_dim, scale=2)

        self.GBC_8 = GBC(8, norm_type='IN')
        self.GN_C = nn.GroupNorm(num_channels=embedding_dim * 4, num_groups=max(1, embedding_dim * 4 // 16))
        #self.GBC_C = GBC(embedding_dim * 4, use_eca=False)

        if fusion_mode == 'original':
            self.use_original = True
            self.GBC_C = GBC(embedding_dim * 4, use_eca=True)
            self.linear_fuse = BottConv(embedding_dim * 4, embedding_dim, embedding_dim // 8, kernel_size=1)
            if attention_type is not None:
                print(f"[Warning] attention_type='{attention_type}' is ignored when fusion_mode='original'")
        elif fusion_mode in ['weighted', 'dynamic']:
            self.use_original = False
            if fusion_mode == 'weighted':
                self.fusion = WeightedFusion(num_scales=4)
            elif fusion_mode == 'dynamic':
                self.fusion = DynamicWeightedFusion(in_channels=embedding_dim, num_scales=4)

            if attention_type == 'eca':
                self.attention = ECA(embedding_dim)
            elif attention_type == 'sfa':
                self.attention = SFA(embedding_dim)
            elif attention_type == 'sebica':
                self.attention = SEBICA(embedding_dim)
            elif attention_type == 'gbc_eca':
                self.attention = GBC(embedding_dim, use_eca=True)
            elif attention_type == 'gbc':
                self.attention = GBC(embedding_dim)
            else:
                raise ValueError(f"Unsupported attention type: {attention_type}")

            self.linear_fuse = BottConv(embedding_dim, embedding_dim, embedding_dim // 8, kernel_size=1)
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        self.dropout = nn.Dropout(p=0.1)
        self.linear_pred = BottConv(embedding_dim, 1, 1, kernel_size=1)
        self.linear_pred_1 = nn.Conv2d(1, 1, kernel_size=1)

    #def _transform(self, c, linear, dy):
    #    b, c_, h, w = c.shape
    #    out = linear(c.reshape(b, c_, h * w).permute(0, 2, 1))
    #    return out.permute(0, 2, 1).reshape(b, dy, h, w)
    
    '''def _transform_c4(self, c):
        b, c_, h, w = c.shape
        out = self.linear_c4(c.reshape(b, c_, h * w).permute(0, 2, 1))
        return self.DySample_C_8(out.permute(0, 2, 1).reshape(b, self.embedding_dim, h, w))
    
    def _transform_c1(self, c):
        b, c_, h, w = c.shape
        out = self.linear_c4(c.reshape(b, c_, h * w).permute(0, 2, 1))
        return self.DySample_C_8(out.permute(0, 2, 1).reshape(b, self.embedding_dim, h, w))
    
    def _transform_c3(self, c):
        b, c_, h, w = c.shape
        out = self.linear_c3(c.reshape(b, c_, h * w).permute(0, 2, 1))
        return self.DySample_C_8(out.permute(0, 2, 1).reshape(b, self.embedding_dim, h, w))
    
    def _transform_c2(self, c):
        b, c_, h, w = c.shape
        out = self.linear_c4(c.reshape(b, c_, h * w).permute(0, 2, 1))
        return self.DySample_C_8(out.permute(0, 2, 1).reshape(b, self.embedding_dim, h, w))'''

    
    def forward(self, inputs: List[torch.Tensor]):
        c4, c3, c2, c1 = inputs[0], inputs[1], inputs[2], inputs[3]

        def transform(c, linear, dy):
            b, c_, h, w = c.shape
            out = linear(c.reshape(b, c_, h * w).permute(0, 2, 1))
            return dy(out.permute(0, 2, 1).reshape(b, self.embedding_dim, h, w))

        out_c4 = transform(c4, self.linear_c4, self.DySample_C_8)
        out_c3 = transform(c3, self.linear_c3, self.DySample_C_4)
        out_c2 = transform(c2, self.linear_c2, self.DySample_C_2)
        #out_c4 = self._transform_c4(c4)
        #out_c3 = self._transform_c3(c3)
        #out_c2 = self._transform_c2(c2)
        b, c, h, w = c1.shape
        out_c1 = self.linear_c1(c1.reshape(b, c, h * w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)

        if self.use_original:
            out_c = self.GBC_C(torch.cat([out_c4, out_c3, out_c2, out_c1], dim=1))
        else:
            #features = torch.jit.annotate(List[torch.Tensor], [out_c4, out_c3, out_c2, out_c1])
            #fused = self.fusion(features)
            fused = self.fusion([out_c4, out_c3, out_c2, out_c1])
            out_c = self.attention(fused)

        out_c = self.linear_fuse(out_c)
        out_c = self.dropout(out_c)
        x = self.linear_pred_1(self.linear_pred(out_c))
        return x