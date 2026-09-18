"""
dual_engine.py - Dual-Engine Wrapper for 2D ResNet-Edge MTF
Universal and modular wrapper around ResNetEdgeMTFEngine.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.engine import ResNetEdgeMTFEngine


class DualEngineMTF(ResNetEdgeMTFEngine):
    """
    Backward-compatible wrapper for ResNetEdgeMTFEngine.
    """
    def __init__(self, model_weights_path: str = None, device: str = None):
        super().__init__(model_weights_path=model_weights_path, device=device)
        self.ort_session = None
        self.torch_model = self.model

    def evaluate_roi_dual(self, roi, target_freq_lpmm=50.0, pixel_size_mm=0.00375):
        res = self.analyze_roi_full(roi, target_freq_lpmm=target_freq_lpmm, pixel_size_mm=pixel_size_mm)
        if res is None:
            return None
        c_mtf = res["classical"]["FilteredMTF"] * 100.0 if res["classical"] and res["classical"].get("FilteredMTF") else 0.0
        n_mtf = res["neural"]["NeuralMTFPercent"] if res["neural"] else 0.0
        delta = res["consensus_delta_percent"] if res["consensus_delta_percent"] is not None else abs(n_mtf - c_mtf)
        return {
            "ClassicalMTFPercent": round(c_mtf, 2),
            "NeuralMTFPercent": round(n_mtf, 2),
            "DeltaPercent": round(delta, 2),
            "Consensus": "PASS" if delta < 8.0 else "REVIEW",
            "Classical": res["classical"],
            "Neural": res["neural"]
        }
