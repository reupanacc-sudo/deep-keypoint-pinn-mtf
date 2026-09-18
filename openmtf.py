"""
openmtf.py - Public Python API for 2D ResNet-Edge MTF Regression
"""

from inference.engine import ResNetEdgeMTFEngine
from mtf_core import compute_slanted_edge_mtf
from training.synthetic_edge_generator import generate_slanted_edge

__version__ = "1.0.0"
__all__ = ["ResNetEdgeMTFEngine", "compute_slanted_edge_mtf", "generate_slanted_edge"]


def analyze_edge(crop, target_freq_lpmm=50.0, pixel_size_mm=0.00375):
    """
    Convenience function to analyze a slanted edge in 1 line of code.
    """
    engine = ResNetEdgeMTFEngine()
    return engine.analyze_roi_full(crop, target_freq_lpmm=target_freq_lpmm, pixel_size_mm=pixel_size_mm)
