"""
deep_keypoint.py - Deep Keypoint & Target Detection Network
Performs sub-pixel crosshair & fiducial localization directly from full camera images.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class SpatialSoftArgmax2d(nn.Module):
    """
    Differentiable spatial soft-argmax layer for sub-pixel keypoint coordinate extraction.
    Converts 2D heatmaps (B, K, H, W) directly into continuous coordinates (B, K, 2).
    """
    def __init__(self, temperature: float = 10.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, heatmaps):
        # heatmaps: (B, K, H, W)
        b, k, h, w = heatmaps.shape
        flat = heatmaps.view(b, k, -1)
        weights = F.softmax(flat * self.temperature, dim=-1).view(b, k, h, w)

        # Coordinate grids in normalized [-1, 1]
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=heatmaps.device),
            torch.linspace(-1.0, 1.0, w, device=heatmaps.device),
            indexing="ij"
        )
        x_coords = torch.sum(weights * x_grid, dim=[2, 3])
        y_coords = torch.sum(weights * y_grid, dim=[2, 3])
        coords = torch.stack([x_coords, y_coords], dim=-1)  # (B, K, 2)
        return coords


class DeepKeypointNet(nn.Module):
    """
    Lightweight Keypoint Detection Network with Spatial Soft-Argmax.
    Detects up to 5 chart fiducials (Center, Left, Right, Top, Bottom) in < 2 ms.
    """
    def __init__(self, num_keypoints: int = 5):
        super().__init__()
        self.num_keypoints = num_keypoints

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 128x128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Heatmap prediction head (32x32 resolution)
        self.head = nn.Conv2d(128, num_keypoints, kernel_size=1)
        self.soft_argmax = SpatialSoftArgmax2d(temperature=15.0)

    def forward(self, x):
        # x: (B, 1, 256, 256)
        feat = self.backbone(x)
        heatmaps = self.head(feat)
        coords = self.soft_argmax(heatmaps)  # (B, K, 2) in [-1, 1]
        return coords, heatmaps
