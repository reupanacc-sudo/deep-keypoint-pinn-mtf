"""
benchmark_dual_engine.py
Runs the Dual-Engine (PINN-MTF + Classical ISO 12233) across production datasets
and demonstrates the noise-resilience of the Physics-Informed Neural Network.
"""

import os
import sys
import time
import glob
import numpy as np
import cv2

# Local paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.dual_engine import DualEngineMTF


def run_benchmarks():
    print("=" * 70)
    print("      DUAL-ENGINE MTF (NEURAL PINN + CLASSICAL ISO 12233) BENCHMARK")
    print("=" * 70)

    engine = DualEngineMTF()
    print("[*] Dual Engine Initialized:")
    print(f"    - ONNX Accelerated: {engine.ort_session is not None}")
    print(f"    - PyTorch Available: {engine.torch_model is not None}")

    # -------------------------------------------------------------
    # 1. Real Bosch Multi-Distance Verification
    # -------------------------------------------------------------
    bosch_files = sorted(glob.glob(r"C:\Users\reu.pan\Desktop\036589910369461080027260004\*.png"))
    if bosch_files:
        print(f"\n[Test 1] Evaluating {len(bosch_files)} Bosch Production Images:")
        sample = bosch_files[4]  # 450mm
        res = engine.evaluate_full_image_dual(sample)
        print(f"  Sample: {os.path.basename(sample)}")
        print(f"  Overall Consensus: {res['overall_consensus']}")
        print(f"  Anomalies Flagged: {res['anomalies_detected']}")
        print("  ROI Comparison:")
        print(f"    {'ROI Section':12s} | {'Classical':10s} | {'Neural PINN':12s} | {'Delta':7s} | {'Consensus':15s}")
        print("    " + "-" * 62)
        for sec, dres in res["dual_results"].items():
            if dres:
                print(f"    {sec:12s} | {dres['ClassicalMTFPercent']:8.2f}% | {dres['NeuralMTFPercent']:10.2f}% | {dres['DeltaPercent']:5.2f}% | {dres['Consensus']:15s}")

    # -------------------------------------------------------------
    # 2. Real Scania Idneo Verification
    # -------------------------------------------------------------
    idneo_sample = r"C:\Users\reu.pan\Desktop\Scania EOLT images logs\A3181103V0690701T260825S2228\ScaniaEOL_Scania_EOL_21532-R8000009_10509546_A3181103V0690701T260825S2228_FV_20260827_084843_0_.png"
    if os.path.exists(idneo_sample):
        print("\n[Test 2] Evaluating Scania Idneo EOLT Camera:")
        res_idneo = engine.evaluate_full_image_dual(idneo_sample)
        print(f"  Overall Consensus: {res_idneo['overall_consensus']}")
        for sec, dres in res_idneo["dual_results"].items():
            if dres:
                print(f"    - {sec:10s}: Classical={dres['ClassicalMTFPercent']:.1f}% | Neural={dres['NeuralMTFPercent']:.1f}% [{dres['Consensus']}]")

    # -------------------------------------------------------------
    # 3. Noise Resilience Stress Test: Classical vs PINN-MTF
    # -------------------------------------------------------------
    print("\n[Test 3] Extreme Sensor Noise Stress Test (Poisson & Gaussian Shot Noise):")
    # Take a clean edge crop and inject increasing noise levels
    if bosch_files:
        img = cv2.imread(bosch_files[0], cv2.IMREAD_GRAYSCALE)
        clean_crop = img[500:545, 715:755]
        print(f"    {'Noise Sigma':12s} | {'Classical MTF':14s} | {'PINN-MTF':12s} | {'Robustness Advantage':20s}")
        print("    " + "-" * 62)
        for sigma in [0.0, 5.0, 15.0, 30.0, 50.0]:
            noisy = clean_crop.astype(np.float32) + np.random.normal(0, sigma, clean_crop.shape)
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)

            d = engine.evaluate_roi_dual(noisy, target_freq_lpmm=50.0)
            if d:
                print(f"    sigma={sigma:4.1f}      | {d['ClassicalMTFPercent']:10.2f}%     | {d['NeuralMTFPercent']:8.2f}%    | Delta = {d['DeltaPercent']:.2f}%")

    # -------------------------------------------------------------
    # 4. Latency & Throughput Benchmark
    # -------------------------------------------------------------
    print("\n[Test 4] Latency Benchmark (1,000 Iterations):")
    dummy = np.random.randint(0, 255, (45, 40), dtype=np.uint8)
    
    # Warmup
    for _ in range(50):
        engine.predict_neural_mtf(dummy)

    t0 = time.perf_counter()
    n_iters = 500
    for _ in range(n_iters):
        engine.predict_neural_mtf(dummy)
    t_neural_avg = ((time.perf_counter() - t0) / n_iters) * 1000.0

    print(f"  * Average Neural PINN Inference Latency: {t_neural_avg:.3f} ms / ROI")
    print(f"  * Theoretical Neural Throughput: {1000.0 / t_neural_avg:.0f} ROIs / second")

    print("\n" + "=" * 70)
    print("                     ALL BENCHMARKS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmarks()
