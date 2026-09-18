"""
train_pinn_mtf.py
Trains the ResNet-Edge PINN model on physics-accurate optical datasets
and exports the trained model to PyTorch (.pt) and ONNX (.onnx).
"""

import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Add local path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from training.synthetic_edge_generator import generate_batch
from models.pinn_mtf import ResNetEdgePINN, PINNLoss


def train_pinn_model(
    epochs: int = 25,
    batch_size: int = 64,
    train_samples: int = 4000,
    val_samples: int = 1000,
    lr: float = 1e-3,
    save_dir: str = None
):
    print("=" * 65)
    print("      TRAINING PHYSICS-INFORMED NEURAL MTF REGRESSOR (PINN)")
    print("=" * 65)

    if save_dir is None:
        save_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device}")

    # 1. Generate Synthetic Optical Datasets
    print(f"[*] Synthesizing {train_samples} training & {val_samples} validation edge patches...")
    t0 = time.time()
    train_rois, freqs, train_gt = generate_batch(train_samples)
    val_rois, _, val_gt = generate_batch(val_samples)
    print(f"[+] Dataset generation completed in {time.time() - t0:.2f}s")

    # Normalize image inputs to [0, 1] float32 tensors: (N, 1, H, W)
    x_train = torch.tensor(train_rois[:, None, :, :], dtype=torch.float32) / 255.0
    y_train = torch.tensor(train_gt, dtype=torch.float32)

    x_val = torch.tensor(val_rois[:, None, :, :], dtype=torch.float32) / 255.0
    y_val = torch.tensor(val_gt, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    # 2. Instantiate Model, Optimizer & Physics-Informed Loss
    model = ResNetEdgePINN(num_outputs=64).to(device)
    criterion = PINNLoss(w_data=1.0, w_dc=0.8, w_mono=0.3, w_smooth=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Training Loop
    print("\n[*] Starting Optimization...")
    best_val_loss = float("inf")
    pt_path = os.path.join(save_dir, "pinn_mtf_weights.pt")
    onnx_path = os.path.join(save_dir, "pinn_mtf.onnx")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss, _ = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss, _ = criterion(preds, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "freqs": freqs,
                "val_loss": val_loss
            }, pt_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

    print(f"\n[+] Optimization Finished. Best Val Loss: {best_val_loss:.5f}")
    print(f"[+] Saved PyTorch Model: {pt_path}")

    # 4. Export to High-Speed ONNX
    print("[*] Exporting Model to ONNX for Sub-Millisecond Industrial Inference...")
    model.eval()
    dummy_input = torch.randn(1, 1, 45, 40, device=device)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["edge_roi"],
            output_names=["mtf_curve"],
            dynamic_axes={"edge_roi": {0: "batch_size"}, "mtf_curve": {0: "batch_size"}},
            opset_version=14
        )
        print(f"[+] Successfully Exported ONNX: {onnx_path}")
    except Exception as e:
        print(f"[!] ONNX export warning: {e}")

    return model, freqs


if __name__ == "__main__":
    train_pinn_model()
