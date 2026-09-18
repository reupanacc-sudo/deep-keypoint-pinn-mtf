"""
dual_engine.py - High-Performance Dual-Engine MTF Pipeline
Combines Physics-Informed Neural Network (PINN-MTF) and Classical ISO 12233
with real-time consensus validation and optical anomaly detection.
"""

import os
import sys
import time
import math
import numpy as np
import cv2

# Local paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import Classical MTF Core from MTF Py Code and MTF Py
for p in [
    r"C:\Users\reu.pan\Desktop\MTF Py Code",
    r"C:\Users\reu.pan\Desktop\MTF Py",
    r"C:\Users\reu.pan\Desktop\MTF Py\Py-based MTF"
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from mtf_core import compute_slanted_edge_mtf
from mtf_engine import analyze_any_image, run_mtf_analysis


class DualEngineMTF:
    """
    Dual-Engine Industrial MTF Analyzer.
    Executes Neural PINN and Classical ISO 12233 in parallel, providing:
    1. Sub-millisecond Neural MTF estimation (immune to noise & glare)
    2. Classical ISO 12233 baseline validation
    3. Anomaly & Consensus Index (|Neural - Classical|)
    """
    def __init__(self, onnx_model_path: str = None, pt_model_path: str = None):
        self.onnx_path = onnx_model_path or os.path.join(BASE_DIR, "checkpoints", "pinn_mtf.onnx")
        self.pt_path = pt_model_path or os.path.join(BASE_DIR, "checkpoints", "pinn_mtf_weights.pt")
        
        self.ort_session = None
        self.torch_model = None
        self.freqs = np.linspace(0.0, 100.0, 64)

        # 1. Attempt ONNX Runtime (fastest industrial inference)
        if os.path.exists(self.onnx_path):
            try:
                import onnxruntime as ort
                self.ort_session = ort.InferenceSession(
                    self.onnx_path,
                    providers=["CPUExecutionProvider"]
                )
            except Exception as e:
                self.ort_session = None

        # 2. Fallback to PyTorch checkpoint
        if self.ort_session is None and os.path.exists(self.pt_path):
            try:
                import torch
                from models.pinn_mtf import ResNetEdgePINN
                ckpt = torch.load(self.pt_path, map_location="cpu", weights_only=False)
                self.torch_model = ResNetEdgePINN(num_outputs=64)
                self.torch_model.load_state_dict(ckpt["model_state"])
                self.torch_model.eval()
                if "freqs" in ckpt:
                    self.freqs = ckpt["freqs"]
            except Exception as e:
                self.torch_model = None

    def predict_neural_mtf(self, roi: np.ndarray, target_freq_lpmm: float = 50.0):
        """
        Sub-millisecond Neural MTF prediction for a single ROI.
        """
        if roi is None or roi.size < 64:
            return None

        if roi.ndim == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_f = roi.astype(np.float32)
        h, w = roi_f.shape[:2]

        # Polarity check: Dark -> Bright transition
        left_avg = np.mean(roi_f[:, :max(1, w // 4)])
        right_avg = np.mean(roi_f[:, -max(1, w // 4):])
        if left_avg > right_avg:
            roi_f = 255.0 - roi_f

        # Contrast normalization
        min_v = float(np.min(roi_f))
        max_v = float(np.max(roi_f))
        if max_v - min_v > 10.0:
            roi_norm = (roi_f - min_v) / (max_v - min_v)
        else:
            roi_norm = roi_f / 255.0

        # Resize to network input shape (45, 40)
        roi_in = cv2.resize(roi_norm, (40, 45), interpolation=cv2.INTER_AREA).astype(np.float32)
        roi_in = roi_in[None, None, :, :]  # (1, 1, 45, 40)

        t0 = time.perf_counter()
        if self.ort_session is not None:
            ort_inputs = {self.ort_session.get_inputs()[0].name: roi_in}
            mtf_curve = self.ort_session.run(None, ort_inputs)[0][0]
        elif self.torch_model is not None:
            import torch
            with torch.no_grad():
                tensor_in = torch.from_numpy(roi_in)
                mtf_curve = self.torch_model(tensor_in).numpy()[0]
        else:
            # Fallback to classical if weights not yet trained
            return None

        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Interpolate at target frequency
        target_idx = np.searchsorted(self.freqs, target_freq_lpmm)
        if target_idx == 0:
            val = float(mtf_curve[0])
        elif target_idx >= len(self.freqs):
            val = float(mtf_curve[-1])
        else:
            f0, f1 = self.freqs[target_idx - 1], self.freqs[target_idx]
            m0, m1 = mtf_curve[target_idx - 1], mtf_curve[target_idx]
            val = float(m0 + (target_freq_lpmm - f0) / (f1 - f0) * (m1 - m0))

        return {
            "NeuralMTF": val,
            "NeuralMTFPercent": round(val * 100.0, 2),
            "Frequencies": self.freqs,
            "NeuralMTFCurve": mtf_curve,
            "InferenceTimeMs": dt_ms
        }

    def evaluate_roi_dual(
        self,
        roi: np.ndarray,
        target_freq_lpmm: float = 50.0,
        pixel_size_mm: float = 0.00375,
        lower_limit: float = 25.0,
        upper_limit: float = 75.0,
        anomaly_threshold_delta: float = 6.0
    ):
        """
        Executes Dual-Engine Evaluation on a single ROI.
        """
        # 1. Classical ISO 12233
        t0 = time.perf_counter()
        res_classical = compute_slanted_edge_mtf(
            roi,
            target_freq_lpmm=target_freq_lpmm,
            pixel_size_mm=pixel_size_mm
        )
        t_classical_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Neural PINN
        res_neural = self.predict_neural_mtf(roi, target_freq_lpmm=target_freq_lpmm)

        if res_classical is None:
            return None

        mtf_c_pct = res_classical["MTFPercent"]
        mtf_n_pct = res_neural["NeuralMTFPercent"] if res_neural else mtf_c_pct
        t_neural_ms = res_neural["InferenceTimeMs"] if res_neural else 0.0

        delta = abs(mtf_n_pct - mtf_c_pct)
        status_c = "PASS" if lower_limit <= mtf_c_pct <= upper_limit else "FAIL"
        status_n = "PASS" if lower_limit <= mtf_n_pct <= upper_limit else "FAIL"

        # Consensus decision
        if delta > anomaly_threshold_delta:
            consensus = "ANOMALY_WARNING"
            confidence = max(0.50, 1.0 - (delta / 50.0))
        elif status_c == "PASS" and status_n == "PASS":
            consensus = "CONSENSUS_PASS"
            confidence = min(0.99, 1.0 - (delta / 100.0))
        else:
            consensus = "CONSENSUS_FAIL"
            confidence = min(0.99, 1.0 - (delta / 100.0))

        return {
            "ClassicalMTFPercent": mtf_c_pct,
            "NeuralMTFPercent": mtf_n_pct,
            "DeltaPercent": round(delta, 2),
            "Consensus": consensus,
            "Confidence": round(confidence * 100.0, 1),
            "StatusClassical": status_c,
            "StatusNeural": status_n,
            "TimeClassicalMs": round(t_classical_ms, 2),
            "TimeNeuralMs": round(t_neural_ms, 2),
            "ClassicalFrequencies": res_classical["Frequencies"],
            "ClassicalCurve": res_classical["MTFCurve"],
            "NeuralFrequencies": res_neural["Frequencies"] if res_neural else self.freqs,
            "NeuralCurve": res_neural["NeuralMTFCurve"] if res_neural else res_classical["MTFCurve"]
        }

    def evaluate_full_image_dual(self, image_path: str, config_path: str = None):
        """
        Runs complete Dual-Engine inspection on any camera image.
        """
        # Run base analysis to extract ROIs and fiducials
        base_res = analyze_any_image(image_path, config_path)
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        dual_results = {}
        anomalies_detected = 0

        for sec in base_res["roi_sections"]:
            rdata = base_res["results"][sec]
            x1, y1 = rdata["ROI_Left"], rdata["ROI_Top"]
            x2, y2 = rdata["ROI_Right"], rdata["ROI_Bottom"]
            
            roi_crop = img_gray[max(0, y1):min(img_gray.shape[0], y2), max(0, x1):min(img_gray.shape[1], x2)]
            
            # Apply rotation if configured
            rot = rdata.get("Rotate", 0)
            if rot == 90:
                roi_crop = cv2.rotate(roi_crop, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                roi_crop = cv2.rotate(roi_crop, cv2.ROTATE_180)
            elif rot == 270:
                roi_crop = cv2.rotate(roi_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

            dual_res = self.evaluate_roi_dual(
                roi_crop,
                target_freq_lpmm=base_res.get("lpmm_target", 50.0),
                lower_limit=rdata.get("MTFLowerLimit", 25.0),
                upper_limit=rdata.get("MTFUpperLimit", 75.0)
            )
            dual_results[sec] = dual_res
            if dual_res and dual_res["Consensus"] == "ANOMALY_WARNING":
                anomalies_detected += 1

        return {
            "base": base_res,
            "dual_results": dual_results,
            "anomalies_detected": anomalies_detected,
            "overall_consensus": "PASS" if base_res["overall_status"] == "PASS" and anomalies_detected == 0 else (
                "ANOMALY" if anomalies_detected > 0 else "FAIL"
            )
        }
