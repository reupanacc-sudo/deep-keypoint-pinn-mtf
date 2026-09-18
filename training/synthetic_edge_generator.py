"""
synthetic_edge_generator.py
Generates physics-accurate slanted edge images and ground truth MTF curves
for training the Physics-Informed Neural Network (PINN-MTF).
"""

import math
import numpy as np
import cv2


def generate_slanted_edge(
    width: int = 40,
    height: int = 45,
    angle_deg: float = 5.0,
    blur_sigma: float = 1.2,
    dark_level: float = 30.0,
    bright_level: float = 220.0,
    noise_sigma: float = 1.5,
    pixel_size_mm: float = 0.00375,
    oversample: int = 8
):
    """
    Synthesizes a realistic optical slanted-edge ROI with known blur (PSF) and ground truth MTF.
    """
    # High-resolution grid for subpixel edge generation
    hr_w = width * oversample
    hr_h = height * oversample

    y_grid, x_grid = np.indices((hr_h, hr_w))
    center_x = hr_w / 2.0
    center_y = hr_h / 2.0

    # Line equation: (x - cx) * cos(theta) + (y - cy) * sin(theta) = distance
    rad = math.radians(angle_deg)
    dist = (x_grid - center_x) * math.cos(rad) + (y_grid - center_y) * math.sin(rad)

    # Ideal step edge
    edge_ideal = np.where(dist >= 0, bright_level, dark_level).astype(np.float64)

    # Apply optical Gaussian blur (PSF)
    hr_sigma = blur_sigma * oversample
    k_size = int(math.ceil(hr_sigma * 6)) | 1
    blurred = cv2.GaussianBlur(edge_ideal, (k_size, k_size), hr_sigma)

    # Downsample back to sensor pixel resolution
    sensor_roi = cv2.resize(blurred, (width, height), interpolation=cv2.INTER_AREA)

    # Add sensor noise (Gaussian readout + Poisson photon shot noise)
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, sensor_roi.shape)
        sensor_roi += noise

    sensor_roi = np.clip(sensor_roi, 0, 255).astype(np.uint8)

    # Calculate Ground Truth MTF curve from Gaussian PSF optics
    # For Gaussian PSF with sigma, MTF(f) = exp(-2 * (pi * sigma * f_px)^2)
    num_freqs = 64
    max_freq_lpmm = 100.0  # up to 100 lp/mm
    freqs_lpmm = np.linspace(0, max_freq_lpmm, num_freqs)
    
    # Convert frequency from lp/mm to cycles/pixel: f_px = f_lpmm * pixel_size_mm
    freqs_px = freqs_lpmm * pixel_size_mm
    gt_mtf = np.exp(-2.0 * (np.pi * blur_sigma * freqs_px) ** 2)

    return sensor_roi, freqs_lpmm, gt_mtf


def generate_batch(batch_size: int = 1000, width: int = 40, height: int = 45):
    """
    Generates a diverse batch of synthetic slanted edges with varying MTF qualities.
    """
    rois = []
    mtf_curves = []
    freq_grid = None

    for _ in range(batch_size):
        # Sample realistic physical parameters
        ang = np.random.uniform(-10.0, 10.0)
        # blur_sigma between 0.35 (very sharp, ~85% MTF) and 3.5 (very blurry, ~10% MTF)
        sigma = np.random.uniform(0.35, 3.2)
        dark = np.random.uniform(15.0, 50.0)
        bright = np.random.uniform(170.0, 240.0)
        noise = np.random.uniform(0.5, 4.0)

        roi, freqs, mtf = generate_slanted_edge(
            width=width,
            height=height,
            angle_deg=ang,
            blur_sigma=sigma,
            dark_level=dark,
            bright_level=bright,
            noise_sigma=noise
        )
        if freq_grid is None:
            freq_grid = freqs

        rois.append(roi)
        mtf_curves.append(mtf)

    return np.array(rois, dtype=np.uint8), freq_grid, np.array(mtf_curves, dtype=np.float32)
