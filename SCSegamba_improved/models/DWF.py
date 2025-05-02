import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, features):
        """
        Args:
            features (list of torch.Tensor): List of feature maps from different scales,
                                             each of shape (B, C, H, W)
        Returns:
            torch.Tensor: Fused feature map of shape (B, C, H, W)
        """
        weights = []
        for i, feature in enumerate(features):
            weight = self.weight_layers[i](feature)  # Shape: (B, 1, 1, 1)
            weights.append(weight)
        # Stack weights and normalize across scales
        stacked_weights = torch.stack(weights, dim=0)  # Shape: (num_scales, B, 1, 1, 1)
        normalized_weights = F.softmax(stacked_weights, dim=0)
        # Apply weights to corresponding features and sum
        fused = sum(w * f for w, f in zip(normalized_weights, features))
        return fused