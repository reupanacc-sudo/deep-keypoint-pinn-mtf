"""
engine.py - 2D ResNet-Edge Sub-Millisecond MTF Regression Engine
Fast, universal, and completely reusable for any camera image or slanted-edge crop.
"""

import os
import sys
import time
import math
import numpy as np
import cv2
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.pinn_mtf import ResNetEdgePINN
from mtf_core import compute_slanted_edge_mtf


class ResNetEdgeMTFEngine:
    """
    Universal 2D ResNet-Edge Convolutional Model for Sub-Millisecond MTF Regression.
    Accepts any arbitrary slanted-edge ROI crop or full camera image.
    """
    def __init__(self, model_weights_path: str = None, device: str = None):
        self.weights_path = model_weights_path or os.path.join(BASE_DIR, "checkpoints", "pinn_mtf_weights.pt")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = ResNetEdgePINN(num_outputs=64)
        self.freqs = np.linspace(0.0, 100.0, 64)
        self.is_loaded = False

        if os.path.exists(self.weights_path):
            try:
                ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=False)
                if "model_state" in ckpt:
                    self.model.load_state_dict(ckpt["model_state"])
                else:
                    self.model.load_state_dict(ckpt)
                self.model.to(self.device)
                self.model.eval()
                if isinstance(ckpt, dict) and "freqs" in ckpt:
                    self.freqs = ckpt["freqs"]
                self.is_loaded = True
            except Exception as e:
                print(f"[!] Warning: Could not load weights ({e}). Initializing model in eval mode.")
                self.model.to(self.device)
                self.model.eval()
        else:
            self.model.to(self.device)
            self.model.eval()

    def preprocess_roi(self, roi: np.ndarray, target_size=(40, 45)):
        """
        Prepares any slanted edge crop for the 2D ResNet-Edge model:
        1. Grayscale conversion
        2. Dark -> Bright polarity normalization
        3. Min-max contrast normalization
        4. Spatial resizing to network input shape (45, 40)
        """
        if roi is None or roi.size < 32:
            return None, None

        if roi.ndim == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        gray_f = gray.astype(np.float32)
        h, w = gray_f.shape[:2]

        # Polarity check: ensure dark is on the left, bright is on the right
        left_avg = float(np.mean(gray_f[:, :max(1, w // 4)]))
        right_avg = float(np.mean(gray_f[:, -max(1, w // 4):]))
        if left_avg > right_avg:
            gray_f = 255.0 - gray_f

        # Contrast normalization
        min_v = float(np.min(gray_f))
        max_v = float(np.max(gray_f))
        if max_v - min_v > 10.0:
            norm = (gray_f - min_v) / (max_v - min_v)
        else:
            norm = gray_f / 255.0

        # Resize to network dimensions (W, H) = (40, 45)
        tensor_in = cv2.resize(norm, target_size, interpolation=cv2.INTER_AREA).astype(np.float32)
        tensor_in = tensor_in[None, None, :, :]  # (1, 1, 45, 40)
        return torch.from_numpy(tensor_in).to(self.device), gray

    def predict_neural_mtf(self, roi: np.ndarray, target_freq_lpmm: float = 50.0):
        """
        Executes sub-millisecond 2D ResNet-Edge neural MTF regression.
        """
        tensor_in, gray_roi = self.preprocess_roi(roi)
        if tensor_in is None:
            return None

        t0 = time.perf_counter()
        with torch.no_grad():
            mtf_curve = self.model(tensor_in).cpu().numpy()[0]
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Enforce physical DC normalization MTF(0) = 1.0 and monotonic lower bound
        mtf_curve[0] = 1.0
        mtf_curve = np.clip(mtf_curve, 0.0, 1.0)

        # Interpolate at target evaluation frequency
        target_idx = np.searchsorted(self.freqs, target_freq_lpmm)
        if target_idx == 0:
            target_val = float(mtf_curve[0])
        elif target_idx >= len(self.freqs):
            target_val = float(mtf_curve[-1])
        else:
            f0, f1 = self.freqs[target_idx - 1], self.freqs[target_idx]
            m0, m1 = mtf_curve[target_idx - 1], mtf_curve[target_idx]
            target_val = float(m0 + (target_freq_lpmm - f0) / (f1 - f0) * (m1 - m0))

        # Calculate Neural MTF50 (frequency where MTF drops to 50%)
        mtf50_freq = self._compute_mtf50(self.freqs, mtf_curve)

        return {
            "NeuralMTF": target_val,
            "NeuralMTFPercent": round(target_val * 100.0, 2),
            "NeuralMTF50": mtf50_freq,
            "Frequencies": self.freqs,
            "NeuralMTFCurve": mtf_curve,
            "InferenceTimeMs": round(dt_ms, 3)
        }

    def analyze_roi_full(
        self,
        roi: np.ndarray,
        target_freq_lpmm: float = 50.0,
        pixel_size_mm: float = 0.00375
    ):
        """
        Performs full dual analysis on any slanted edge crop:
        1. Sub-millisecond 2D ResNet-Edge neural regression
        2. Classical ISO 12233 4-stage physics pipeline (ESF -> LSF -> FFT -> MTF)
        """
        if roi is None or roi.size < 64:
            return None

        if roi.ndim == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        # 1. Neural ResNet-Edge Regression
        neural_res = self.predict_neural_mtf(gray, target_freq_lpmm=target_freq_lpmm)

        # 2. Classical ISO 12233 Physics Decomposition
        classical_res = compute_slanted_edge_mtf(
            gray,
            target_freq_lpmm=target_freq_lpmm,
            pixel_size_mm=pixel_size_mm
        )

        # 3. Compute Consensus Metric
        consensus_delta = None
        if neural_res and classical_res and classical_res.get("FilteredMTF") is not None:
            c_val = classical_res["FilteredMTF"]
            n_val = neural_res["NeuralMTF"]
            consensus_delta = round(abs(n_val - c_val) * 100.0, 2)

        return {
            "gray_crop": gray,
            "neural": neural_res,
            "classical": classical_res,
            "consensus_delta_percent": consensus_delta,
            "target_freq_lpmm": target_freq_lpmm
        }

    def auto_detect_slanted_edges(
        self,
        full_image: np.ndarray,
        max_edges: int = 12,
        roi_size: tuple = (50, 50)
    ):
        """
        Universal slanted edge detector for ANY camera image.
        Finds high-contrast edge patches with slant angles suitable for ISO 12233 analysis.
        """
        if full_image is None or full_image.size < 1000:
            return []

        if full_image.ndim == 3:
            gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = full_image.copy()

        h, w = gray.shape
        rw, rh = roi_size

        # Compute gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)

        # Non-maximum suppression grid search for edge candidates
        step = max(rw, rh) // 2
        candidates = []

        for y in range(rh // 2, h - rh, step):
            for x in range(rw // 2, w - rw, step):
                patch_mag = mag[y:y+rh, x:x+rw]
                avg_grad = float(np.mean(patch_mag))
                std_grad = float(np.std(patch_mag))

                # Check if region contains a strong edge with high variation
                if avg_grad > 25.0 and std_grad > 15.0:
                    candidates.append((avg_grad, (x, y, x + rw, y + rh)))

        # Sort candidates by gradient strength and pick top diverse locations
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected_rois = []
        min_dist_sq = (rw * 1.5) ** 2

        for _, (x1, y1, x2, y2) in candidates:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            too_close = False
            for _, sx1, sy1, sx2, sy2 in selected_rois:
                scx = (sx1 + sx2) / 2.0
                scy = (sy1 + sy2) / 2.0
                if (cx - scx)**2 + (cy - scy)**2 < min_dist_sq:
                    too_close = True
                    break
            if not too_close:
                name = f"ROI_{len(selected_rois) + 1}"
                selected_rois.append((name, x1, y1, x2, y2))
                if len(selected_rois) >= max_edges:
                    break

        return selected_rois

    def _compute_mtf50(self, freqs, mtf_curve):
        """Calculates MTF50 frequency via linear interpolation."""
        try:
            below_idx = np.where(mtf_curve <= 0.5)[0]
            if len(below_idx) == 0:
                return round(float(freqs[-1]), 1)
            idx = below_idx[0]
            if idx == 0:
                return 0.0
            f0, f1 = freqs[idx - 1], freqs[idx]
            m0, m1 = mtf_curve[idx - 1], mtf_curve[idx]
            if abs(m1 - m0) < 1e-6:
                return round(float(f0), 1)
            f50 = f0 + (0.5 - m0) / (m1 - m0) * (f1 - f0)
            return round(float(f50), 1)
        except Exception:
            return 0.0
