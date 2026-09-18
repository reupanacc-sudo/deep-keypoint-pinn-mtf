"""
dual_overlay_visualizer.py
Generates visual diagnostic overlays comparing Static ROI vs Dynamic 50/50 AI ROI
and plots the comparative MTF curves.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def generate_dual_overlay_image(img_gray: np.ndarray, comparisons: dict, mode: str = "both"):
    """
    Renders the image with Static (Red/Orange) vs Dynamic 50/50 (Cyan/Green) ROI overlays.

    Parameters:
        img_gray: Grayscale full camera image.
        comparisons: Dict returned from dynamic_roi_engine.process_full_image_static_vs_dynamic.
        mode: 'static', 'dynamic', or 'both'.

    Returns:
        np.ndarray: BGR image with bounding box annotations and luminance balance badges.
    """
    if img_gray.ndim == 2:
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    else:
        overlay = img_gray.copy()

    for sec, comp in comparisons.items():
        sx1, sy1, sx2, sy2 = comp["static_rect"]
        dx1, dy1, dx2, dy2 = comp["dynamic_rect"]
        s_bal = comp["static_balance"]
        d_bal = comp["dynamic_balance"]

        # 1. Draw Static ROI (Red/Coral)
        if mode in ("static", "both"):
            cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), (0, 0, 255), 1, cv2.LINE_AA)
            s_txt = f"{sec} [Static {s_bal['dark_pct']:.0f}/{s_bal['bright_pct']:.0f}%]"
            cv2.putText(
                overlay,
                s_txt,
                (sx1, max(15, sy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 50, 255),
                1,
                cv2.LINE_AA
            )

        # 2. Draw Dynamic 50/50 AI ROI (Cyan/Neon Green)
        if mode in ("dynamic", "both"):
            cv2.rectangle(overlay, (dx1, dy1), (dx2, dy2), (255, 255, 0), 2, cv2.LINE_AA)
            d_txt = f"{sec} [AI 50/50]"
            cv2.putText(
                overlay,
                d_txt,
                (dx1, min(overlay.shape[0] - 8, dy2 + 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 0),
                1,
                cv2.LINE_AA
            )

    return overlay


def plot_static_vs_dynamic_curves(comparison_data: dict, output_plot_path: str = None):
    """
    Generates a high-resolution publication-quality plot comparing
    Static ROI MTF vs Dynamic 50/50 AI MTF curves along with ESF baselines.
    """
    sec = comparison_data["ROIName"]
    s_mtf = comparison_data["static_mtf"]
    d_mtf = comparison_data["dynamic_mtf"]
    s_bal = comparison_data["static_balance"]
    d_bal = comparison_data["dynamic_balance"]

    fig, (ax_mtf, ax_esf) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    fig.patch.set_facecolor("#121417")

    # 1. MTF Curve Plot
    ax_mtf.set_facecolor("#0d1117")
    ax_mtf.grid(True, linestyle="--", alpha=0.3, color="#30363d")

    if s_mtf:
        f_s = s_mtf["Frequencies"]
        m_s = s_mtf["MTFCurve"]
        mask_s = f_s <= 100.0
        ax_mtf.plot(
            f_s[mask_s],
            m_s[mask_s],
            label=f"Static ROI ({s_bal['dark_pct']:.0f}%/{s_bal['bright_pct']:.0f}% Lum) - MTF: {s_mtf['MTFPercent']:.1f}%",
            color="#ff4444",
            linestyle="--",
            linewidth=2.0
        )

    if d_mtf:
        f_d = d_mtf["Frequencies"]
        m_d = d_mtf["MTFCurve"]
        mask_d = f_d <= 100.0
        ax_mtf.plot(
            f_d[mask_d],
            m_d[mask_d],
            label=f"AI Dynamic 50/50 Centered - MTF: {d_mtf['MTFPercent']:.1f}%",
            color="#00ffff",
            linewidth=2.5
        )

    ax_mtf.axvline(x=50.0, color="#8a99a8", linestyle=":", label="Target Frequency (50 lp/mm)")
    ax_mtf.set_title(f"ROI '{sec}' MTF: Static vs Dynamic 50/50 AI", color="#e6edf3", fontsize=11, fontweight="bold")
    ax_mtf.set_xlabel("Spatial Frequency (lp/mm)", color="#8a99a8")
    ax_mtf.set_ylabel("MTF (0.0 to 1.0)", color="#8a99a8")
    ax_mtf.tick_params(colors="#8a99a8")
    ax_mtf.set_ylim(0, 1.05)
    ax_mtf.set_xlim(0, 100)
    ax_mtf.legend(facecolor="#181a1f", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8.5)

    # 2. Edge Spread Function (ESF) Comparison Plot
    ax_esf.set_facecolor("#0d1117")
    ax_esf.grid(True, linestyle="--", alpha=0.3, color="#30363d")

    if s_mtf and "EdgeSpreadFunction" in s_mtf:
        esf_s = s_mtf["EdgeSpreadFunction"]
        ax_esf.plot(esf_s, color="#ff4444", linestyle="--", label="Static ESF (Truncated Baseline)", alpha=0.8)

    if d_mtf and "EdgeSpreadFunction" in d_mtf:
        esf_d = d_mtf["EdgeSpreadFunction"]
        ax_esf.plot(esf_d, color="#00ffff", linewidth=2.0, label="Dynamic 50/50 ESF (Symmetric)")

    ax_esf.set_title("Edge Spread Function (ESF) Centering", color="#e6edf3", fontsize=11, fontweight="bold")
    ax_esf.set_xlabel("Super-Sampled Sub-Pixel Bins (0.25 px)", color="#8a99a8")
    ax_esf.set_ylabel("Intensity (0 - 255)", color="#8a99a8")
    ax_esf.tick_params(colors="#8a99a8")
    ax_esf.legend(facecolor="#181a1f", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8.5)

    plt.tight_layout()

    if output_plot_path:
        os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
        plt.savefig(output_plot_path, dpi=120, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        return output_plot_path
    else:
        plt.close(fig)
        return fig
