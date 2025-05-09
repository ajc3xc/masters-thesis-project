import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class DynamicWeightedFusion(nn.Module):
    def __init__(self, in_channels, num_scales):
        super(DynamicWeightedFusion, self).__init__()
        self.in_channels = in_channels
        self.num_scales = num_scales
        # Learnable parameters to compute weights for each scale
        self.weight_layers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.Sigmoid()
            ) for _ in range(num_scales)
        ])

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list of torch.Tensor): List of feature maps from different scales,
                                             each of shape (B, C, H, W)
        Returns:
            torch.Tensor: Fused feature map of shape (B, C, H, W)
        """
        weights = []
        #for i, feature in enumerate(features):
        #    weight = self.weight_layers[i](feature)  # Shape: (B, 1, 1, 1)
        #    weights.append(weight)
        weights = [
            self.weight_layers[0](features[0]),
            self.weight_layers[1](features[1]),
            self.weight_layers[2](features[2]),
            self.weight_layers[3](features[3])
        ]



        # Stack weights and normalize across scales
        stacked_weights = torch.stack(weights, dim=0)  # Shape: (num_scales, B, 1, 1, 1)
        normalized_weights = F.softmax(stacked_weights, dim=0)

        w0 = normalized_weights[0]
        w1 = normalized_weights[1]
        w2 = normalized_weights[2]
        w3 = normalized_weights[3]

        f0, f1, f2, f3 = features[0], features[1], features[2], features[3]
        fused = w0 * f0 + w1 * f1 + w2 * f2 + w3 * f3

        # Apply weights to corresponding features and sum
        #fused = sum(w * f for w, f in zip(normalized_weights, features))
        return fused