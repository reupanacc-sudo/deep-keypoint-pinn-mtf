"""
resnet_balancer.py - 2D ResNet-Edge Model for Subpixel 50/50 Luminance Equalization
Takes any uncentered ROI image patch and predicts the exact subpixel drift Delta_x
and dark/bright luminance distribution to ensure a perfect 50/50 transition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))


class ResNet5050Balancer(nn.Module):
    """
    2D ResNet-Edge model for subpixel edge localization and 50/50 luminance equalization.
    """
    def __init__(self):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ConvResidualBlock(32, 64, stride=2)   # (H/2, W/2)
        self.layer2 = ConvResidualBlock(64, 128, stride=2)  # (H/4, W/4)
        self.layer3 = ConvResidualBlock(128, 128, stride=2) # (H/8, W/8)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dual prediction heads:
        # Head 1: Subpixel drift Delta_x (in pixels, [-20, +20])
        self.fc_drift = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        # Head 2: Luminance balance ratio [dark_ratio, bright_ratio] in [0, 1]
        self.fc_balance = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x: (B, 1, 45, 40)
        feat = self.in_conv(x)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.pool(feat)
        flat = torch.flatten(feat, 1)

        drift = self.fc_drift(flat)       # (B, 1)
        balance = self.fc_balance(flat)   # (B, 2)
        return drift, balance
