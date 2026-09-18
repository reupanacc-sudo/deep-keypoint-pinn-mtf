"""
train_balancer.py
Trains the ResNet5050Balancer model to predict subpixel edge drift Delta_x
and luminance balance ratios from uncentered ROI crops.
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.resnet_balancer import ResNet5050Balancer
from training.synthetic_edge_generator import generate_slanted_edge


def generate_drift_dataset(num_samples: int = 5000, width: int = 40, height: int = 45):
    """
    Generates synthetic optical edges with random subpixel center offsets Delta_x.
    """
    rois = []
    drifts = []
    balances = []

    for _ in range(num_samples):
        # Random edge angle and blur
        ang = np.random.uniform(-8.0, 8.0)
        sigma = np.random.uniform(0.5, 2.5)
        dark = np.random.uniform(20.0, 50.0)
        bright = np.random.uniform(180.0, 240.0)
        noise = np.random.uniform(0.5, 3.0)

        # Base edge (width + 30 so we can crop with arbitrary shift)
        pad_w = width + 40
        raw_roi, _, _ = generate_slanted_edge(
            width=pad_w,
            height=height,
            angle_deg=ang,
            blur_sigma=sigma,
            dark_level=dark,
            bright_level=bright,
            noise_sigma=noise
        )

        # Shift the crop window: shift_px between -15 and +15 px
        shift_px = np.random.uniform(-14.0, 14.0)
        center_x = pad_w / 2.0 + shift_px
        x1 = int(round(center_x - width / 2.0))
        x1 = max(0, min(pad_w - width, x1))
        crop = raw_roi[:, x1:x1 + width]

        # Calculate exact ground truth drift in the crop:
        # If edge is at crop_x = (pad_w/2 - x1), drift from center is:
        edge_crop_x = (pad_w / 2.0) - x1
        actual_drift = edge_crop_x - (width / 2.0)

        dark_ratio = np.clip(edge_crop_x / width, 0.05, 0.95)
        bright_ratio = 1.0 - dark_ratio

        # Resize to (45, 40)
        crop_norm = cv2.resize(crop, (width, height)).astype(np.float32) / 255.0

        rois.append(crop_norm)
        drifts.append(actual_drift)
        balances.append([dark_ratio, bright_ratio])

    x_tensor = torch.tensor(np.array(rois)[:, None, :, :], dtype=torch.float32)
    y_drift = torch.tensor(np.array(drifts)[:, None], dtype=torch.float32)
    y_bal = torch.tensor(np.array(balances), dtype=torch.float32)
    return x_tensor, y_drift, y_bal


def train_balancer(epochs: int = 20, batch_size: int = 64):
    print("=" * 65)
    print("    TRAINING 2D RESNET-EDGE 50/50 LUMINANCE BALANCER")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on: {device}")

    # Generate Datasets
    print("[*] Generating synthetic edge crops with random drift offsets...")
    x_train, y_drift_train, y_bal_train = generate_drift_dataset(4000)
    x_val, y_drift_val, y_bal_val = generate_drift_dataset(1000)

    train_loader = DataLoader(TensorDataset(x_train, y_drift_train, y_bal_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_drift_val, y_bal_val), batch_size=batch_size, shuffle=False)

    model = ResNet5050Balancer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_drift = nn.SmoothL1Loss()
    criterion_bal = nn.MSELoss()

    best_val_loss = float("inf")
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "resnet_balancer_weights.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, b_drift, b_bal in train_loader:
            bx, b_drift, b_bal = bx.to(device), b_drift.to(device), b_bal.to(device)
            optimizer.zero_grad()
            pred_drift, pred_bal = model(bx)
            loss = criterion_drift(pred_drift, b_drift) + 2.0 * criterion_bal(pred_bal, b_bal)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, b_drift, b_bal in val_loader:
                bx, b_drift, b_bal = bx.to(device), b_drift.to(device), b_bal.to(device)
                pred_drift, pred_bal = model(bx)
                loss = criterion_drift(pred_drift, b_drift) + 2.0 * criterion_bal(pred_bal, b_bal)
                val_loss += loss.item() * bx.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "val_loss": val_loss}, ckpt_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

    print(f"[+] Balancer Training Complete. Saved model: {ckpt_path}")
    return model


if __name__ == "__main__":
    train_balancer()
