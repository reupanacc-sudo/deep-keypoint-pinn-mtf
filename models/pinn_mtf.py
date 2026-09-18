"""
pinn_mtf.py - Physics-Informed Neural Network for Direct Edge-to-MTF Curve Regression
Embeds Fourier optics physical constraints directly into the network loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
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


class ResNetEdgePINN(nn.Module):
    """
    Compact ResNet-Edge architecture optimized for sub-millisecond edge-to-MTF inference.
    """
    def __init__(self, num_outputs: int = 64):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ConvBlock(32, 64, stride=2)   # (H/2, W/2)
        self.layer2 = ConvBlock(64, 128, stride=2)  # (H/4, W/4)
        self.layer3 = ConvBlock(128, 128, stride=2) # (H/8, W/8)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_outputs),
            nn.Sigmoid()  # Guarantees output values are strictly bounded in [0, 1]
        )

    def forward(self, x):
        # x shape: (B, 1, H, W) normalized to [0, 1]
        feat = self.in_conv(x)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)
        mtf_curve = self.fc(feat)
        return mtf_curve


class PINNLoss(nn.Module):
    """
    Composite Physics-Informed Fourier Optics Loss Function.
    Combines data fidelity with fundamental laws of optics:
    1. MSE Data Loss
    2. DC Normalization: MTF(f=0) == 1.0
    3. Monotonicity Regularization: dMTF/df <= 0 for non-spurious physical response
    4. Curvature Regularization: smooth second derivative
    """
    def __init__(self, w_data=1.0, w_dc=0.5, w_mono=0.2, w_smooth=0.1):
        super().__init__()
        self.w_data = w_data
        self.w_dc = w_dc
        self.w_mono = w_mono
        self.w_smooth = w_smooth

    def forward(self, y_pred, y_true):
        # 1. Supervised Data Loss
        l_data = F.mse_loss(y_pred, y_true)

        # 2. Physics Constraint 1: DC component at f=0 must equal 1.0
        dc_val = y_pred[:, 0]
        l_dc = torch.mean((dc_val - 1.0) ** 2)

        # 3. Physics Constraint 2: Monotonic Decreasing Penalty
        # First difference along frequency axis: diff = y[i+1] - y[i]
        diff_1 = y_pred[:, 1:] - y_pred[:, :-1]
        # Any positive difference violates physics in diffraction-limited/defocused optical systems
        l_mono = torch.mean(F.relu(diff_1) ** 2)

        # 4. Physics Constraint 3: Smoothness Regularization (2nd derivative)
        diff_2 = diff_1[:, 1:] - diff_1[:, :-1]
        l_smooth = torch.mean(diff_2 ** 2)

        total_loss = (
            self.w_data * l_data +
            self.w_dc * l_dc +
            self.w_mono * l_mono +
            self.w_smooth * l_smooth
        )
        return total_loss, {
            "l_data": l_data.item(),
            "l_dc": l_dc.item(),
            "l_mono": l_mono.item(),
            "l_smooth": l_smooth.item(),
            "total": total_loss.item()
        }
