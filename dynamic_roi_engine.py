"""
dynamic_roi_engine.py
Implements Static ROI vs AI Dynamic 50/50 Luminance Centering.
Accepts input images with static priors and dynamically centers the ROI to ensure
a perfect 50/50 luminance distribution across the slanted edge.
"""

import os
import sys
import math
import numpy as np
import cv2

# Local paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

for p in [
    r"C:\Users\reu.pan\Desktop\MTF Py Code",
    r"C:\Users\reu.pan\Desktop\MTF Py",
    r"C:\Users\reu.pan\Desktop\MTF Py\Py-based MTF"
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from mtf_core import compute_slanted_edge_mtf
from mtf_engine import analyze_any_image


def analyze_luminance_balance(roi_crop: np.ndarray):
    """
    Measures the luminance transition, subpixel edge location, and dark/bright balance ratio.
    """
    if roi_crop is None or roi_crop.size < 64:
        return None

    if roi_crop.ndim == 3:
        gray = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_crop.copy()

    h, w = gray.shape
    gray_f = gray.astype(np.float64)

    # Detect polarity
    left_avg = np.mean(gray_f[:, :max(1, w // 4)])
    right_avg = np.mean(gray_f[:, -max(1, w // 4):])
    inverted = False
    if left_avg > right_avg:
        gray_f = 255.0 - gray_f
        inverted = True

    # Find subpixel edge centroid on each line
    kernel = np.array([-0.5, 0.0, 0.5], dtype=np.float64).reshape(1, 3)
    grad_x = cv2.filter2D(gray_f, cv2.CV_64F, kernel)
    edge_x_list = []

    for r in range(h):
        row_grad = grad_x[r, :]
        max_idx = int(np.argmax(row_grad))
        if 1 <= max_idx < w - 1:
            y1, y2, y3 = row_grad[max_idx - 1], row_grad[max_idx], row_grad[max_idx + 1]
            denom = (y1 - 2 * y2 + y3)
            if abs(denom) > 1e-6:
                delta = np.clip(0.5 * (y1 - y3) / denom, -1.0, 1.0)
                edge_x_list.append(max_idx + delta)
            else:
                edge_x_list.append(float(max_idx))

    if len(edge_x_list) > h * 0.3:
        avg_edge_x = float(np.median(edge_x_list))
    else:
        avg_edge_x = float((w - 1) / 2.0)

    # Desired center is (w - 1) / 2.0
    desired_center_x = (w - 1) / 2.0
    drift_px = avg_edge_x - desired_center_x

    # Calculate Dark vs Bright width percentage using continuous coordinates [0, w]
    left_pct = ((avg_edge_x + 0.5) / w) * 100.0
    right_pct = 100.0 - left_pct

    if not inverted:
        dark_pct, bright_pct = left_pct, right_pct
    else:
        dark_pct, bright_pct = right_pct, left_pct

    return {
        "avg_edge_x": avg_edge_x,
        "drift_px": drift_px,
        "dark_pct": round(dark_pct, 1),
        "bright_pct": round(bright_pct, 1),
        "is_inverted": inverted
    }


def compute_static_vs_dynamic_roi(
    img: np.ndarray,
    static_rect: tuple,
    target_freq_lpmm: float = 50.0,
    pixel_size_mm: float = 0.00375,
    rotate_deg: int = 0
):
    """
    Compares Static ROI vs Dynamic 50/50 AI Centered ROI on a single edge.

    Parameters:
        img: Full 2D grayscale camera image.
        static_rect: (x1, y1, x2, y2) static nominal bounding box.
        target_freq_lpmm: Evaluation spatial frequency.
        pixel_size_mm: Sensor pixel pitch.
        rotate_deg: ROI orientation rotation (0, 90, 180, 270).

    Returns:
        dict: {
            "static_rect": (x1, y1, x2, y2),
            "dynamic_rect": (dx1, dy1, dx2, dy2),
            "static_crop": np.ndarray,
            "dynamic_crop": np.ndarray,
            "static_mtf": dict,
            "dynamic_mtf": dict,
            "static_balance": dict,
            "dynamic_balance": dict,
            "correction_shift_px": float
        }
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img.shape

    x1, y1, x2, y2 = static_rect
    w = x2 - x1
    h = y2 - y1

    # 1. Extract Static Crop
    static_crop = img[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)].copy()
    if rotate_deg == 90:
        static_crop_rot = cv2.rotate(static_crop, cv2.ROTATE_90_CLOCKWISE)
    elif rotate_deg == 180:
        static_crop_rot = cv2.rotate(static_crop, cv2.ROTATE_180)
    elif rotate_deg == 270:
        static_crop_rot = cv2.rotate(static_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        static_crop_rot = static_crop

    # Measure Static Balance
    static_bal = analyze_luminance_balance(static_crop_rot)

    # 2. Dynamic 50/50 AI Centering
    drift = static_bal["drift_px"] if static_bal else 0.0

    # In original unrotated image coordinates:
    if rotate_deg == 0:
        dx_shift = int(round(drift))
        dy_shift = 0
    elif rotate_deg == 90:
        dx_shift = 0
        dy_shift = -int(round(drift))
    elif rotate_deg == 180:
        dx_shift = -int(round(drift))
        dy_shift = 0
    elif rotate_deg == 270:
        dx_shift = 0
        dy_shift = int(round(drift))
    else:
        dx_shift = int(round(drift))
        dy_shift = 0

    dx1 = max(0, min(img_w - w, x1 + dx_shift))
    dy1 = max(0, min(img_h - h, y1 + dy_shift))
    dx2 = dx1 + w
    dy2 = dy1 + h
    dynamic_rect = (dx1, dy1, dx2, dy2)

    # Extract Dynamic Crop with Subpixel Continuous Centering
    crop_w, crop_h = static_crop_rot.shape[1], static_crop_rot.shape[0]
    M_subpixel = np.float32([[1, 0, -drift], [0, 1, 0]])
    dynamic_crop_rot = cv2.warpAffine(
        static_crop_rot,
        M_subpixel,
        (crop_w, crop_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT
    )

    # Measure Dynamic Balance
    dynamic_bal = analyze_luminance_balance(dynamic_crop_rot)

    # 3. Compute MTF Curves for Both
    static_mtf = compute_slanted_edge_mtf(
        static_crop_rot,
        target_freq_lpmm=target_freq_lpmm,
        pixel_size_mm=pixel_size_mm
    )
    dynamic_mtf = compute_slanted_edge_mtf(
        dynamic_crop_rot,
        target_freq_lpmm=target_freq_lpmm,
        pixel_size_mm=pixel_size_mm
    )

    return {
        "static_rect": static_rect,
        "dynamic_rect": dynamic_rect,
        "static_crop": static_crop_rot,
        "dynamic_crop": dynamic_crop_rot,
        "static_mtf": static_mtf,
        "dynamic_mtf": dynamic_mtf,
        "static_balance": static_bal,
        "dynamic_balance": dynamic_bal,
        "correction_shift_px": round(abs(drift), 1)
    }


def process_full_image_static_vs_dynamic(image_path: str, config_path: str = None):
    """
    Evaluates an entire camera image with both Static and Dynamic 50/50 AI ROIs.
    """
    import configparser
    base_res = analyze_any_image(image_path, config_path)
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    cfg = None
    if config_path and os.path.isfile(config_path):
        cfg = configparser.ConfigParser()
        cfg.read(config_path)

    comparison_results = {}
    for sec in base_res["roi_sections"]:
        rdata = base_res["results"][sec]
        static_rect = (
            rdata["ROI_Left"],
            rdata["ROI_Top"],
            rdata["ROI_Right"],
            rdata["ROI_Bottom"]
        )
        rot = 0
        if cfg and cfg.has_section(sec) and cfg.has_option(sec, "Rotate"):
            rot = cfg.getint(sec, "Rotate", fallback=0)
        else:
            rot = rdata.get("Rotate", 0)

        comp = compute_static_vs_dynamic_roi(
            img_gray,
            static_rect,
            target_freq_lpmm=base_res.get("lpmm_target", 50.0),
            rotate_deg=rot
        )
        comp["ROIName"] = sec
        comparison_results[sec] = comp

    return {
        "base": base_res,
        "comparisons": comparison_results
    }
