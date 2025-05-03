import torch
import torch.nn as nn

class WeightedFusion(nn.Module):
    """Learnable fusion of multi-scale features"""
    def __init__(self, num_scales):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_scales))
    def forward(self, features):
        w = torch.softmax(self.weights, dim=0)
        return sum(f * w[i] for i, f in enumerate(features))